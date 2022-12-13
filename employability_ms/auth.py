from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from .models import *

_auth = Blueprint('_auth', __name__)

@_auth.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@_auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('.index'))
