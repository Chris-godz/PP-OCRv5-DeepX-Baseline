cd $PWD
python3 -m venv venv
source venv/bin/activate

python -m pip install -r ./requirements.txt

cd $HOME/dx-all-suite/dx-runtime/dx_rt
./build.sh --clean

