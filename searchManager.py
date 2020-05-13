# ............IMPORTANT.......... #

# 1. Create project from PyCharm as a Flask project directly (automatically handle dependency)
# 2. Go to Project 'Settings' set Python interpreter as 3.7.x
# 3. Install SQLAlchemy from the project dependency in PyCharm
# 4. Install Flask-sqlalchemy from the project dependency in PyCharm
# 5. Run this following two lines in the python console to create the database
#   >>> from searchManager import db
#   >>> db.create_all()

# ...................................... #


import os
import time

from flask import g
from flask import Flask
from flask import render_template
from flask import request
from flask import redirect,flash
from flask_sqlalchemy import SQLAlchemy
import code_search
import data_preprocess
from sqlalchemy import create_engine


from flask import url_for, send_from_directory
# import flask_whooshalchemy as wa

# ============ This is for Database Confiiguration ============= #
project_dir = os.path.dirname(os.path.abspath(__file__))
database_file = "sqlite:///{}".format(os.path.join(project_dir, "source_code_database.db"))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = database_file
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
# app.config["WHOOSH_BASE"] = "whoosh"

db = SQLAlchemy(app)

db.Model.metadata.reflect(db.engine)


class Employee(db.Model):

    #__searchable__ =['name']
    # --------- Add as many column as you want --------- #
    __tablename__ = 'employee'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    name = db.Column(db.String(80), unique=False, nullable=False, primary_key=False)

    def __repr__(self):
        return "<Id: {}>".format(self.id), "<Name: {}>".format(self.name)

# wa.whoosh_index(app, Employee)


class Book_List(db.Model):

    # ----- this maps already created database table to read/SELECT ----- #
    __tablename__ = 'book'
    __table_args__ = {'extend_existing': True}
    # id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    # title = db.Column(db.String(80), unique=False, nullable=False, primary_key=False)

    def __repr__(self):
        return "<Id: {}>".format(self.id), "<Title: {}>".format(self.title)


class Tb_Function(db.Model):

    #__searchable__ =['name']
    # --------- Add as many column as you want --------- #
    __tablename__ = 'tb_function'
    __table_args__ = {'extend_existing': True}

    # id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
    # name = db.Column(db.String(80), unique=False, nullable=False, primary_key=False)
    id = db.Column(db.Integer, unique=True, nullable=True, primary_key=True)
    function_name = db.Column(db.String(100), unique=False, nullable=False, primary_key=False)
    function_body = db.Column(db.String(500), unique=False, nullable=False, primary_key=False)
    function_descriptor = db.Column(db.String(500), unique=False, nullable=False, primary_key=False)

    # def __repr__(self):
    #     return "<Id: {}>".format(self.id), "<Name: {}>".format(self.function_name)
class Tb_Query(db.Model):
    __tablename__ = 'tb_query'
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.Integer, unique=True, nullable=True, primary_key=True)
    query = db.Column(db.String(500), unique=False, nullable=False, primary_key=False)
    length = db.Column(db.Numeric, unique=False, nullable=False, primary_key=False)
    time_in_second = db.Column(db.Numeric, unique=False, nullable=False, primary_key=False)
    length_of_words = db.Column(db.Numeric, unique=False, nullable=False, primary_key=False)


@app.route('/search', methods=["GET", "POST"])
def search_info():

    # function_descriptor_pair_list=data_preprocess.get_pair_list()
    #
    # print(function_descriptor_pair_list)
    #
    # try:
    #     for i in function_descriptor_pair_list:
    #         row = Tb_Function(function_name=i[0],function_body=i[1],function_descriptor=i[2])
    #         db.session.add(row)
    #     db.session.commit()
    # except Exception as e:
    #     print("Failed to add row")
    #     print(e)
    # functions = Tb_Function.query.all()
    return render_template("search.html")#, functions=functions)


@app.before_request
def before_request():
    g.request_start_time = time.time()
    g.request_time = lambda: "%.5fs" % (time.time() - g.request_start_time)


@app.route('/result', methods=["GET", "POST"])
def result():
    code_snippets = "No Search Query Provided"
    if request.form:
        try:
            search_query = request.form.get("search_query")
            functions = Tb_Function.query.all()
            code_snippets, sorted_query_with_sim = code_search.code_snippet_result(functions, search_query)
            code_snippets = code_snippets.split('\n')
            # print("search: ", code_snippets)
            db.session.commit()
        except Exception as e:
            print("Failed to find Related Code")
            print(e)

    # return redirect("/search")
    try:
        # saving query in tb_query
        search_query = request.form.get("search_query")
        len_words_arr = search_query.split()
        len_words = len(len_words_arr)
        exec_time = g.request_time().split("s")
        row = Tb_Query(query=search_query, length=len(search_query),time_in_second=exec_time[0],length_of_words=len_words)
        db.session.add(row)
        db.session.commit()
    except Exception as e:
        pass
    results = [sorted_query_with_sim, code_snippets]
    # print(results[0])
    return render_template("result.html", result=results)


# ------------------------------------------------ code for upload
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'py', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# --- --- This checks correct file or not ------------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/file')
def file_home():
    return render_template("upload.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        upload_message = ''
        try:
            # ---------- This is the file object -------------- #
            file = request.files['file']

            if file and allowed_file(file.filename):
                #filename = secure_filename(file.filename)
                filename = file.filename

                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pair_list = data_preprocess.get_pair_list_for_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # print(pair_list)

                # try:
                #     for i in pair_list:
                #         row = Tb_Function(function_name=i[0],function_body=i[1],function_descriptor=i[2])
                #         try:
                #             db.session.add(row)
                #             db.session.commit()
                #         except Exception as e:
                #             print(i[0],"function was added before in the database")
                #
                # except Exception as e:
                #     print("Failed to add row")
                #     print(e)

                for i in pair_list:
                    row = Tb_Function(function_name=i[0], function_body=i[1], function_descriptor=i[2])
                    try:
                        db.session.add(row)
                        db.session.commit()
                        upload_message = 'File Uploaded Successfully'
                    except Exception as e:
                        upload_message += "*"
                        upload_message += i[0]
                        upload_message += " function was uploaded before in the database \n"
                        # flash(upload_message)
                        # print(upload_message)
                upload_message = upload_message.split('\n')
                # , upload_message = upload_message
                return render_template("upload_message.html", upload_message=upload_message)
                # return redirect(url_for('uploaded_file', filename=filename, upload_message=upload_message))  # This reads the file and shows in the browser #
            else:
                upload_message = "Not an uploadable file"
                print("Not an uploadable file")

        except Exception as e:
            upload_message = "Couldn't upload file"
            print("Couldn't upload file")
            print(e)
            return redirect('/file')

    return redirect('/file')


@app.route('/upload/<filename>')
def uploaded_file(filename ):
    return render_template("upload_message.html", upload_message="sonet")
    # return send_from_directory(app.config['UPLOAD_FOLDER'], filename) # takes the file from storage and shows its content


##------------------------------------------------


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
    # app.run(debug=True)



