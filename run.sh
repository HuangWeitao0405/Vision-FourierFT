python3 main.py --config=./exps/sdlora_inr.json  >> InR.log 2>&1 &
# python3 main.py --config=./exps/sdlora_ina.json  >> InA.log 2>&1 &
python3 main.py --config=./exps/sdfourierft_inr.json  >> FourierFT.log 2>&1 &
python3 main.py --config=./exps/sdlora_c100.json  >> CF100.log 2>&1 &
python3 main.py --config=./exps/sdfourierft_c100.json  >> CF100FourierFT.log 2>&1 &