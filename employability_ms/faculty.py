from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, make_response, session
from flask_login import login_required, current_user
from .models import *
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from sqlalchemy import delete

_faculty = Blueprint('_faculty', __name__)

@_faculty.route('/login_register_faculty', methods=['GET'])
def login_registerFaculty_view():
    auth_user=current_user
    if auth_user.is_authenticated:
        if auth_user.user_type == -1 or auth_user.user_type == 0:
            return redirect(url_for('.faculty_dashboard'))
        else:
            return redirect(url_for('_auth.index'))
            
    return render_template("Faculty/login_admin.html")

@_faculty.route('/login_faculty', methods=['GET', 'POST'])
def login_faculty():
    auth_user=current_user
    if auth_user.is_authenticated:
        if auth_user.user_type == -1 or auth_user.user_type == 0:
            return redirect(url_for('.faculty_dashboard'))
        else:
            return redirect(url_for('_auth.index'))
    else:
        if request.method == 'POST':
            user = User.query.filter_by(email=request.form['email'], department='faculty').first()
            print(user)
            if user:
                if check_password_hash(user.password, request.form['password']):
                    login_user(user, remember=True)
                    return redirect(url_for('.faculty_dashboard'))
                else:
                    flash('Invalid or wrong password', category='error')
            else:
                flash('No record found', category='error')
    return redirect(url_for('.login_registerFaculty_view'))

@_faculty.route('/faculty_dashboard', methods=['GET'])
@login_required
def faculty_dashboard():
    auth_user=current_user
    if auth_user.user_type == -1 or auth_user.user_type == 0:
        auth_user=current_user
        page = request.args.get('page', 1, type=int)
        students_record = db.session.query(User, PredictionResult).join(PredictionResult).filter(User.user_type == 1).paginate(page=page, per_page=10)# fetch user students only
    else:
        return redirect(url_for('_auth.index'))
    
    return render_template("Faculty/facultyEnd.html", auth_user=auth_user, students_record=students_record)

@_faculty.route('/delete_results', methods=['POST'])
@login_required
def delete_results():
    auth_user=current_user
    # delete_result = delete(User).where(User.id == request.form['user_id'])
    delete_pred_result = delete(PredictionResult).where(PredictionResult.result_id == request.form['user_id'])
    
    db.session.execute(delete_pred_result)
    # db.session.execute(delete_result)
    db.session.commit()
    return redirect(url_for('.faculty_dashboard'))