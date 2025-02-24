import tkinter as tk
from tkinter import ttk
import cv2
import threading
import csv
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter
from PIL import Image, ImageTk
import numpy as np
import time

class OpenBCIWebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema OpenBCI + Webcam")
        self.root.geometry("1100x650")
        
        # Frame do ICC (Sinais do OpenBCI)
        self.icc_frame = tk.Frame(root, width=600, height=400, bg='white')
        self.icc_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10)

        # Widget de Texto para os sinais EEG
        self.text_output = tk.Text(self.icc_frame, height=20, width=70, wrap="word")
        self.text_output.pack(pady=10, padx=10)
        self.text_output.insert(tk.END, "Aguardando dados do OpenBCI...\n")

        # Frame do Indicador de Concentração
        self.focus_frame = tk.Frame(root, width=250, height=200, bg='lightgray')
        self.focus_frame.grid(row=0, column=1, padx=10, pady=10)
        
        self.focus_label = tk.Label(self.focus_frame, text="Concentração", font=("Arial", 14, "bold"))
        self.focus_label.pack()

        self.focus_canvas = tk.Canvas(self.focus_frame, width=100, height=100, bg="white")
        self.focus_canvas.pack()
        self.focus_circle = self.focus_canvas.create_oval(10, 10, 90, 90, fill="blue")
        
        self.focus_text = tk.Label(self.focus_frame, text="Aguardando...", font=("Arial", 12))
        self.focus_text.pack()

        # Frame da Câmera
        self.cam_frame = tk.Frame(root, width=250, height=200, bg='gray')
        self.cam_frame.grid(row=1, column=1, padx=10, pady=10)
        self.cam_label = tk.Label(self.cam_frame)
        self.cam_label.pack()
        
        # Frame dos Botões
        self.btn_frame = tk.Frame(root, width=250, height=200, bg='lightgray')
        self.btn_frame.grid(row=2, column=1, padx=10, pady=10)
        
        # Botões
        self.start_btn = ttk.Button(self.btn_frame, text="Iniciar", command=self.start_stream)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = ttk.Button(self.btn_frame, text="Parar", command=self.stop_stream)
        self.stop_btn.pack(pady=5)
        
        self.quit_btn = ttk.Button(self.btn_frame, text="Sair", command=self.on_close)
        self.quit_btn.pack(pady=5)
        
        # Inicialização da Câmera e Gravação de Vídeo
        self.cap = cv2.VideoCapture(0)
        self.video_writer = None  # Inicialmente não há gravação de vídeo
        self.running = False
        self.update_camera()
        
        # Configuração do OpenBCI (via BrainFlow Streamer)
        self.board = None
        self.setup_openbci()

        # Criar arquivo CSV para salvar os dados
        self.csv_file = open("eeg_data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Tempo", "Frame", "Delta", "Theta", "Alpha", "Beta", "Gamma", "Concentracao"])

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
            self.start_video_recording()
            self.update_openbci()  # Atualiza a cada 1 segundo
    
    def stop_stream(self):
        if self.running:
            self.running = False
            self.board.stop_stream()
            self.board.release_session()
            self.stop_video_recording()
            self.text_output.insert(tk.END, "Captura encerrada.\n")
            self.text_output.see(tk.END)

    def start_video_recording(self):
        """ Inicia a gravação do vídeo da webcam """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec de vídeo
        fps = 20  # Frames por segundo
        frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))  # Resolução da câmera
        self.video_writer = cv2.VideoWriter("video.avi", fourcc, fps, frame_size)

    def stop_video_recording(self):
        """ Finaliza a gravação do vídeo """
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print("Vídeo salvo como video.avi")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Adiciona o tempo ao frame
            timestamp_text = time.strftime('%H:%M:%S')
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, timestamp_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

            # Salvar frame no vídeo se a gravação estiver ativa
            if self.running and self.video_writer is not None:
                self.video_writer.write(frame_bgr)
        
        self.root.after(10, self.update_camera)

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

        # Parar gravação de vídeo
        self.stop_video_recording()

        # Liberar a câmera corretamente
        try:
            if self.cap.isOpened():
                self.cap.release()
        except Exception as e:
            print(f"Erro ao liberar câmera: {e}")

        # Fechar a janela do Tkinter
        self.root.quit()
        self.root.destroy()
        print("Aplicação encerrada com sucesso.")

if __name__ == "__main__":
    root = tk.Tk()
    app = OpenBCIWebcamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
