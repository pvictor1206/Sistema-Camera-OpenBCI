# server.py
from flask import Flask, jsonify
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import numpy as np

app = Flask(__name__)

# Configurar BrainFlow para capturar os dados
params = BrainFlowInputParams()
params.ip_address = "127.0.0.1"  # OpenBCI enviando para localhost
params.ip_port = 6677  # Porta configurada no OpenBCI
params.streaming_board = -1  # Capturar de um streamer

board = BoardShim(-1, params)
board.prepare_session()
board.start_stream()

@app.route('/data', methods=['GET'])
def get_eeg_data():
    """Captura dados do OpenBCI e retorna como JSON"""
    try:
        data = board.get_board_data()
        if data.shape[1] > 0:
            eeg_data = data[1:9, :]  # Pegando os 8 primeiros canais de EEG
            avg_signal = np.mean(eeg_data, axis=1).tolist()  # Média por canal

            response = {
                "status": "success",
                "data": avg_signal  # Enviar os valores médios dos 8 canais
            }
        else:
            response = {
                "status": "no_data",
                "message": "Nenhum dado disponível ainda"
            }
    except Exception as e:
        response = {
            "status": "error",
            "message": str(e)
        }
    return jsonify(response)

@app.route('/')
def home():
    return "Servidor OpenBCI rodando. Acesse /data para ver os dados."

def run_server():
    # Desabilita o reloader para evitar que a thread seja iniciada duas vezes
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)

if __name__ == '__main__':
    run_server()
