#!/bin/sh

pip3 install virtualenv
virtualenv env
source env/bin/activate; pip3 install -r bin/requirements.txt

source env/bin/activate;