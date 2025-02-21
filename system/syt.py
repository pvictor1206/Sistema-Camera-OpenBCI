import tkinter as tk
from tkinter import ttk
import cv2
import threading
import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from PIL import Image, ImageTk
import numpy as np
import time

class OpenBCIWebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema OpenBCI + Webcam")
        self.root.geometry("800x600")
        
        # Frame do ICC (Sinais do OpenBCI)
        self.icc_frame = tk.Frame(root, width=500, height=400, bg='white')
        self.icc_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        
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
        
        self.quit_btn = ttk.Button(self.btn_frame, text="Sair", command=root.quit)
        self.quit_btn.pack(pady=5)
        
        # Inicialização da Câmera
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.update_camera()
        
        # Configuração do OpenBCI (Cyton Board)
        self.board = None
        self.openbci_thread = None
        self.setup_openbci()

    def setup_openbci(self):
        params = BrainFlowInputParams()
        params.serial_port = 'COM3'  # Altere para a porta correta
        self.board = BoardShim(0, params)  # 0 = Cyton Board
        
    def start_stream(self):
        if not self.running:
            self.running = True
            self.board.prepare_session()
            self.board.start_stream()
            self.openbci_thread = threading.Thread(target=self.update_openbci)
            self.openbci_thread.start()
    
    def stop_stream(self):
        if self.running:
            self.running = False
            self.board.stop_stream()
            self.board.release_session()
    
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
        while self.running:
            data = self.board.get_board_data()
            eeg_data = data[1:9, :]  # Pegando os 8 primeiros canais de EEG
            avg_signal = np.mean(eeg_data)
            print(f"Média do sinal EEG: {avg_signal}")
            time.sleep(1)
    
    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenBCIWebcamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
