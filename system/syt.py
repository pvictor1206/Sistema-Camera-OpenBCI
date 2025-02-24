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
import sys

# Importe a função que inicia o servidor Flask
from server import run_server


class OpenBCIWebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema OpenBCI + Webcam")
        self.root.geometry("1200x700")

        # **Frame principal**
        self.main_frame = tk.Frame(root, bg="white")
        self.main_frame.pack(fill="both", expand=True)

        # **Botões no canto superior esquerdo**
        self.btn_frame = tk.Frame(self.main_frame, bg='lightgray', relief="ridge", bd=2)
        self.btn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")

        self.start_btn = ttk.Button(self.btn_frame, text="Iniciar", command=self.start_stream)
        self.start_btn.pack(side="left", padx=5, pady=5)

        self.stop_btn = ttk.Button(self.btn_frame, text="Parar", command=self.stop_stream)
        self.stop_btn.pack(side="left", padx=5, pady=5)

        self.quit_btn = ttk.Button(self.btn_frame, text="Sair", command=self.on_close)
        self.quit_btn.pack(side="left", padx=5, pady=5)

        # **Indicador de Concentração ao lado dos botões**
        self.focus_frame = tk.Frame(self.main_frame, width=200, height=100, bg='lightgray', relief="ridge", bd=2)
        self.focus_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        self.focus_label = tk.Label(self.focus_frame, text="Concentracao", font=("Arial", 14, "bold"), bg="lightgray")
        self.focus_label.pack()

        self.focus_canvas = tk.Canvas(self.focus_frame, width=100, height=100, bg="white")
        self.focus_canvas.pack()
        self.focus_circle = self.focus_canvas.create_oval(10, 10, 90, 90, fill="blue")

        self.focus_text = tk.Label(self.focus_frame, text="Aguardando...", font=("Arial", 12), bg="lightgray")
        self.focus_text.pack()

        # **Frame para exibir os dados EEG (Centro da tela)**
        self.icc_frame = tk.Frame(self.main_frame, width=600, height=400, bg='white', relief="ridge", bd=2)
        self.icc_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nw")

        self.text_output = tk.Text(self.icc_frame, height=20, width=70, wrap="word")
        self.text_output.pack(pady=10, padx=10)
        self.text_output.insert(tk.END, "Aguardando dados do OpenBCI...\n")

        # **Exibição da Câmera (Lado direito dos dados)**
        self.cam_frame = tk.Frame(self.main_frame, width=300, height=250, bg='gray', relief="ridge", bd=2)
        self.cam_frame.grid(row=1, column=2, padx=10, pady=10, sticky="ne")
        self.cam_label = tk.Label(self.cam_frame)
        self.cam_label.pack()

        # **Inicialização da Câmera e Gravação de Vídeo**
        self.cap = cv2.VideoCapture(0)
        self.video_writer = None
        self.running = False
        
        # Defina a variável de concentração
        self.concentration = 0.0
        
        self.update_camera()

        # **Configuração do OpenBCI**
        self.board = None

        # **Criar arquivo CSV para salvar os dados**
        self.csv_file = open("eeg_data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Tempo", "Frame", "Delta", "Theta", "Alpha", "Beta", "Gamma", "Concentracao"])

    def start_stream(self):
        if not self.running:
            self.running = True
            # Como os dados vêm do servidor, não precisamos preparar o board localmente.
            self.start_video_recording()
            self.update_openbci()

    def stop_stream(self):
        if self.running:
            self.running = False
            self.stop_video_recording()
            self.text_output.insert(tk.END, "Captura encerrada.\n")
            self.text_output.see(tk.END)

    def start_video_recording(self):
        """Inicia a gravação do vídeo da webcam"""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20
        frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))
        self.video_writer = cv2.VideoWriter("video.avi", fourcc, fps, frame_size)

    def stop_video_recording(self):
        """Finaliza a gravação do vídeo"""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            print("📁 Vídeo salvo como video.avi")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # Converte o frame para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Adiciona o timestamp
            timestamp_text = time.strftime('%H:%M:%S')
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, timestamp_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Adiciona o valor da concentração abaixo do horário
            conc_text = f"Concentracao: {self.concentration:.2f}"
            cv2.putText(frame_bgr, conc_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # Converte para imagem e atualiza o label da câmera
            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
            # Se estiver gravando, escreve o frame
            if self.running and self.video_writer:
                self.video_writer.write(frame_bgr)
        # Atualiza a cada 33ms (~30 FPS)
        self.root.after(33, self.update_camera)

    def fetch_openbci_data_thread(self):
        try:
            import requests
            # Timeout de 1 segundo para evitar bloqueios longos
            response = requests.get("http://localhost:5000/data", timeout=1)
            if response.status_code == 200:
                data = response.json()
            else:
                data = {"status": "error", "message": f"Status code: {response.status_code}"}
        except Exception as e:
            data = {"status": "error", "message": str(e)}
        # Agenda o processamento dos dados na thread principal
        self.root.after(0, self.process_openbci_data, data)

    def process_openbci_data(self, data):
        if data.get("status") == "success":
            avg_signal = data.get("data")
            if not avg_signal or len(avg_signal) < 5:
                self.text_output.insert(tk.END, "⚠️ Dados insuficientes no servidor\n")
            else:
                # Usa os 5 primeiros valores
                delta, theta, alpha, beta, gamma = avg_signal[:5]
                timestamp = time.strftime('%H:%M:%S')
                self.text_output.insert(tk.END, f"\nTempo: {timestamp}\n")
                self.text_output.insert(
                    tk.END, f"Delta: {delta:.5f} | Theta: {theta:.5f} | Alpha: {alpha:.5f} | Beta: {beta:.5f} | Gamma: {gamma:.5f}\n"
                )
                self.text_output.see(tk.END)
                # Calcula o nível de concentração (ajuste a fórmula conforme necessário)
                focus_level = beta / max((alpha + theta + delta), 1e-6)
                self.update_focus_widget(focus_level)
                frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.csv_writer.writerow([timestamp, frame_number, delta, theta, alpha, beta, gamma, focus_level])
        else:
            self.text_output.insert(tk.END, f"⚠️ Erro: {data.get('message', 'Erro desconhecido')}\n")

    def update_openbci(self):
        if self.running:
            threading.Thread(target=self.fetch_openbci_data_thread).start()
            self.root.after(1000, self.update_openbci)

    def update_focus_widget(self, focus_level):
        # Atualiza o widget de concentração com o valor formatado (entre 0 e 1)
        self.focus_text.config(text=f"{focus_level:.2f}")
        # Atualiza também a variável de instância usada na exibição na câmera
        self.concentration = focus_level

    def on_close(self):
        self.running = False
        self.cap.release()
        self.csv_file.close()
        self.root.quit()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    # Inicia o servidor Flask em uma thread separada
    flask_thread = threading.Thread(target=run_server)
    flask_thread.setDaemon(True)
    flask_thread.start()

    root = tk.Tk()
    app = OpenBCIWebcamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
