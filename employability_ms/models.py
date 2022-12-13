from sqlalchemy.sql import func
from sqlalchemy import column, func
from . import db, marsh, app
from flask_login import UserMixin

from marshmallow import Schema, fields

class UserSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'first_name', 'middle_name', 'last_name', 'contact_number', 'email', 'department', 'password', 'user_type')
    
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    middle_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255), nullable=False)
    contact_number = db.Column(db.String(15), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False) 
    department = db.Column(db.String(150), nullable=False) 
    password = db.Column(db.String(255), nullable=False)
    user_type = db.Column(db.SmallInteger, nullable=False, default=1) # -1 Superadmin(Built-in), 0 - Admin, 1 - Personnel
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())
    pred_results = db.relationship('PredictionResult', backref='user', uselist=False)

    def __init__(self, first_name, middle_name, last_name, contact_number, email, department, password, user_type):
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.contact_number = contact_number
        self.email = email 
        self.department = department 
        self.password = password
        self.user_type = user_type


class PredictionResultSchema(marsh.Schema):
    class Meta:
        fields = ('result_id', 'main_rank', 'sub_rank1', 'sub_rank2', 'sub_rank3', 'user_id')

class PredictionResult(db.Model):
    result_id = db.Column(db.Integer, primary_key=True)
    main_rank = db.Column(db.String(255), nullable=False)
    sub_rank1 = db.Column(db.String(255), nullable=False)
    sub_rank2 = db.Column(db.String(255), nullable=False)
    sub_rank3 = db.Column(db.String(15), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())

    def __init__(self, main_rank, sub_rank1, sub_rank2, sub_rank3, user_id):
        self.main_rank = main_rank
        self.sub_rank1 = sub_rank1
        self.sub_rank2 = sub_rank2
        self.sub_rank3 = sub_rank3
        self.user_id = user_id