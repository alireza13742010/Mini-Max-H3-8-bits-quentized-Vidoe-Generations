python download_weights.py --check              # report only, downloads nothing
python download_weights.py                      # verify, fetch what's missing, then verdict
python download_weights.py --localize-index     # also fix the index (see below)
python download_weights.py --workflows ref2va   # ignore the other workflow's transformer folder
###########################################################################################333
to run the model
MINIMAX_H3_ALLOW_DOWNLOAD=1 streamlit run minimax_h3_streamlit_app.py
