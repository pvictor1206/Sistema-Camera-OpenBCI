from flask import Flask, jsonify
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter
import numpy as np
import time

app = Flask(__name__)

def initialize_board():
    params = BrainFlowInputParams()
    params.serial_port = "COM4"      # Substitua pela porta correta do seu dispositivo
    # Para conexão via serial não são necessários ip_address/ip_port.
    board_id = -1  # 2 eletrodos.
    try:
        board = BoardShim(board_id, params)
        time.sleep(5)  # Aguarda o dispositivo inicializar
        board.prepare_session()
        board.start_stream()
        time.sleep(3)  # Aguarda o buffer ser preenchido
        print("Board real inicializado com sucesso.")
    except Exception as e:
        print("Erro ao inicializar board real:", e)
    return board

board = initialize_board()

@app.route('/data', methods=['GET'])
def get_eeg_data():
    """Captura dados do OpenBCI e retorna os valores das bandas (Delta, Theta, Alpha, Beta, Gamma) em formato JSON."""
    try:
        data = board.get_board_data()
        eeg_data = data[1:3, :]
        sampling_rate = BoardShim.get_sampling_rate(board.get_board_id())
        band_powers = DataFilter.get_avg_band_powers(eeg_data, [0, 1], sampling_rate, True)
        # Verifica se o retorno é um número (erro); use np.issubdtype para capturar np.int32, etc.
        if np.issubdtype(type(band_powers), np.integer):
            raise Exception("Erro ao calcular bandas: código " + str(band_powers))
        if not hasattr(band_powers, '__len__'):
            raise Exception("Retorno inesperado de get_avg_band_powers: " + str(band_powers))
        bands = band_powers[0]
        if np.issubdtype(type(bands), np.integer) or not hasattr(bands, '__len__') or len(bands) < 5:
            raise Exception("Formato inesperado dos valores de banda: " + str(bands))
        response = {"status": "success", "data": bands.tolist()}
    except Exception as e:
        response = {"status": "error", "message": str(e)}
    return jsonify(response)

@app.route('/')
def home():
    return "Servidor OpenBCI rodando. Acesse /data para ver os dados."

def run_server():
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)

if __name__ == '__main__':
    run_server()
