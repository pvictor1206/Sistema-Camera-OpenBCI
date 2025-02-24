from flask import Flask, jsonify
from brainflow.board_shim import BoardShim, BrainFlowInputParams
import numpy as np

app = Flask(__name__)

# Configurar BrainFlow para capturar os dados de um dispositivo real
params = BrainFlowInputParams()
params.serial_port = "COM3"      # Substitua "COM3" pela porta correta do seu dispositivo
params.ip_address = "127.0.0.1"    # Se aplicável
params.ip_port = 6677            # Se aplicável

# Utilize o board_id correspondente ao seu dispositivo real (por exemplo, 0 para OpenBCI Cyton)
board_id = -1
board = BoardShim(board_id, params)
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
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)

if __name__ == '__main__':
    run_server()
