#............IMPORTANT..........#

# 1. Create a folder in the project directory 'uploads' to store the files
# 2. Run the program as localhost:5000/file
# 3. Uploaded file will take to automated new url for viewing

#......................................#

import os

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import data_preprocess
from flask_sqlalchemy import SQLAlchemy
import searchManager

#from werkzeug import secure_filename




# project_dir = os.path.dirname(os.path.abspath(__file__))
# database_file = "sqlite:///{}".format(os.path.join(project_dir, "source_code_database.db"))
#
# app = Flask(__name__)
# app.config["SQLALCHEMY_DATABASE_URI"] = database_file
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = True
# #app.config["WHOOSH_BASE"] = "whoosh"
#
# db = SQLAlchemy(app)
#
# db.Model.metadata.reflect(db.engine)


# class Tb_Function(db.Model):
#
#     #__searchable__ =['name']
#     # --------- Add as many column as you want --------- #
#     __tablename__ = 'tb_function'
#     __table_args__ = {'extend_existing': True}
#
#     # id = db.Column(db.Integer, unique=True, nullable=False, primary_key=True)
#     # name = db.Column(db.String(80), unique=False, nullable=False, primary_key=False)
#     id =db.Column(db.Integer, unique=True, nullable=True, primary_key=True)
#     function_name = db.Column(db.String(100), unique=False, nullable=False, primary_key=False)
#     function_body=db.Column(db.String(500), unique=False, nullable=False, primary_key=False)
#     function_descriptor=db.Column(db.String(500), unique=False, nullable=False, primary_key=False)


#-------- This is for the upload storage----------- #
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'py', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
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
        try:
            # ---------- This is the file object -------------- #
            file = request.files['file']

            if file and allowed_file(file.filename):
                #filename = secure_filename(file.filename)
                filename = file.filename

                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                pair_list = data_preprocess.get_pair_list_for_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # print(pair_list)

                try:
                    for i in pair_list:
                        row = searchManager.Tb_Function(function_name=i[0],function_body=i[1],function_descriptor=i[2])
                        print(i[0],"\n")
                        print(i[1], "\n")
                        # print(row)
                        searchManager.db.session.add(row)
                        searchManager.db.session.commit()
                except Exception as e:
                    print("Failed to add row")
                    print(e)



                return redirect(url_for('uploaded_file', filename=filename))  # This reads the file and shows in the browser #
            else:
                print("Not an uploadable file")

        except Exception as e:
            print("Couldn't upload file")
            print(e)
            return redirect('/file')

    return redirect('/file')


@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) # takes the file from storage and shows its content


if __name__ == '__main__':
    app.run(debug=True)