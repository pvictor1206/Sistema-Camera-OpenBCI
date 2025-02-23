import tkinter as tk
from tkinter import ttk
import cv2
import threading
import csv
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from PIL import Image, ImageTk
import numpy as np
import time

class OpenBCIWebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema OpenBCI + Webcam")
        self.root.geometry("1000x600")
        
        # Frame do ICC (Sinais do OpenBCI)
        self.icc_frame = tk.Frame(root, width=600, height=400, bg='white')
        self.icc_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
        # Widget de Texto para os sinais EEG
        self.text_output = tk.Text(self.icc_frame, height=20, width=70, wrap="word")
        self.text_output.pack(pady=10, padx=10)
        self.text_output.insert(tk.END, "Aguardando dados do OpenBCI...\n")

        # Frame da Câmera
        self.cam_frame = tk.Frame(root, width=250, height=200, bg='gray')
        self.cam_frame.grid(row=0, column=1, padx=10, pady=10)
        self.cam_label = tk.Label(self.cam_frame)
        self.cam_label.pack()
        
        # Frame dos Botões
        self.btn_frame = tk.Frame(root, width=250, height=200, bg='lightgray')
        self.btn_frame.grid(row=1, column=1, padx=10, pady=10)
        
        # Botões
        self.start_btn = ttk.Button(self.btn_frame, text="Iniciar", command=self.start_stream)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(self.btn_frame, text="Parar", command=self.stop_stream)
        self.stop_btn.pack(pady=5)
        
        self.quit_btn = ttk.Button(self.btn_frame, text="Sair", command=self.on_close)
        self.quit_btn.pack(pady=5)
        
        # Inicialização da Câmera
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.update_camera()
        
        # Configuração do OpenBCI (via BrainFlow Streamer)
        self.board = None
        self.setup_openbci()

        # Criar arquivo CSV para salvar os dados
        self.csv_file = open("eeg_data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Tempo", "Canal 1", "Canal 2", "Canal 3", "Canal 4", 
                                  "Canal 5", "Canal 6", "Canal 7", "Canal 8"])

    def setup_openbci(self):
        params = BrainFlowInputParams()
        params.ip_address = "127.0.0.1"  # OpenBCI enviando para localhost
        params.ip_port = 6677  # Porta configurada no OpenBCI_GUI
        params.streaming_board = -1  # Indica que estamos pegando dados de um Streamer
        self.board = BoardShim(-1, params)  # Conectar ao BrainFlow Streamer
        
    def start_stream(self):
        if not self.running:
            self.running = True
            self.board.prepare_session()
            self.board.start_stream()
            self.update_openbci()  # Atualiza a cada 1 segundo
    
    def stop_stream(self):
        if self.running:
            self.running = False
            self.board.stop_stream()
            self.board.release_session()
            self.text_output.insert(tk.END, "Captura encerrada.\n")
            self.text_output.see(tk.END)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
        self.root.after(10, self.update_camera)
    
    def update_openbci(self):
        if self.running:
            try:
                data = self.board.get_board_data()
                if data.shape[1] > 0:
                    eeg_data = data[1:9, :]  # Pegando os 8 primeiros canais de EEG
                    
                    # Calcular valores médios para exibição
                    avg_signals = np.mean(eeg_data, axis=1)

                    # Atualizar a interface gráfica
                    self.text_output.insert(tk.END, f"\nTempo: {time.strftime('%H:%M:%S')}\n")
                    for i, signal in enumerate(avg_signals, start=1):
                        self.text_output.insert(tk.END, f"Canal {i}: {signal:.5f} µV\n")
                    self.text_output.see(tk.END)

                    # Salvar os dados no CSV
                    row_data = [time.time()] + avg_signals.tolist()
                    self.csv_writer.writerow(row_data)

                self.root.after(1000, self.update_openbci)  # Atualiza a cada 1 segundo
            except Exception as e:
                self.text_output.insert(tk.END, f"Erro ao obter dados: {e}\n")
    
    def on_close(self):
        """ Fecha a aplicação corretamente """
        self.running = False  # Para os loops de atualização
        
        # Liberar recursos do OpenBCI, se a sessão estiver ativa
        try:
            if self.board is not None:
                self.board.stop_stream()
                self.board.release_session()
        except Exception as e:
            print(f"Erro ao liberar OpenBCI: {e}")

        # Fechar arquivo CSV corretamente
        try:
            self.csv_file.close()
        except Exception as e:
            print(f"Erro ao fechar arquivo CSV: {e}")

        # Liberar a câmera corretamente
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(f"Erro ao liberar câmera: {e}")

        # Fechar a janela do Tkinter e encerrar a aplicação
        self.root.quit()  # Fecha o loop do Tkinter
        self.root.destroy()  # Libera todos os recursos gráficos
        print("Aplicação encerrada com sucesso.")


if __name__ == "__main__":
    root = tk.Tk()
    app = OpenBCIWebcamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
