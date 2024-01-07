#!/bin/bash
pip install virtualenv
virtualenv env --python=python3.12.1
source env/bin/activate
pip install -r requirements.txt
python controller/app.py
