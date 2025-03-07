import tkinter as tk
from tkinter import ttk
import cv2
import threading
import csv
from PIL import Image, ImageTk
import numpy as np
import time
import sys
import pyautogui

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
        
        self.record_btn = ttk.Button(self.btn_frame, text="Gravar Tela", command=self.toggle_screen_recording)
        self.record_btn.pack(side="left", padx=5, pady=5)

        self.quit_btn = ttk.Button(self.btn_frame, text="Sair", command=self.on_close)
        self.quit_btn.pack(side="left", padx=5, pady=5)

        # **Indicador de Concentração e Relaxamento**
        self.focus_frame = tk.Frame(self.main_frame, width=200, height=100, bg='lightgray', relief="ridge", bd=2)
        self.focus_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nw")

        self.focus_label = tk.Label(self.focus_frame, text="Estado Mental", font=("Arial", 14, "bold"), bg="lightgray")
        self.focus_label.pack()

        self.focus_text = tk.Label(self.focus_frame, text="Foco: -- \nRelaxamento: --", font=("Arial", 12), bg="lightgray")
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

        # **Inicialização da Câmera**
        self.cap = cv2.VideoCapture(0)
        self.running = False

        # Variáveis de concentração e relaxamento
        self.concentration = 0.0
        self.relaxation = 0.0

        self.update_camera()

        # **Configuração do OpenBCI** (dados vêm do servidor)
        self.board = None

        # **Criar arquivo CSV para salvar os dados**
        self.csv_file = open("eeg_data.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Atualizei a ordem das colunas:
        self.csv_writer.writerow(["Tempo", "Concentracao", "Relaxamento", "Delta", "Theta", "Alpha", "Beta", "Gamma"])

    def start_stream(self):
        if not self.running:
            self.running = True
            self.start_video_recording()
            self.update_openbci()
            
    def start_video_recording(self):
        """Inicia a gravação do vídeo da webcam"""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20
        frame_size = (int(self.cap.get(3)), int(self.cap.get(4)))
        self.video_writer = cv2.VideoWriter("video_webcan.avi", fourcc, fps, frame_size)
    
    def stop_video_recording(self):
        """Finaliza a gravação do vídeo"""
        if self.video_writer:
            self.video_writer.release()  # Libera o vídeo
            self.video_writer = None
            print("📁 Vídeo salvo como video.avi")  # Mensagem de confirmação
            
    def update_openbci(self):
        if self.running:
            threading.Thread(target=self.fetch_openbci_data_thread).start()
            self.root.after(1000, self.update_openbci)

    def stop_stream(self):
        if self.running:
            self.running = False
            self.text_output.insert(tk.END, "Captura encerrada.\n")
            self.text_output.see(tk.END)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp_text = time.strftime('%H:%M:%S')
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Exibir tempo na tela
            cv2.putText(frame_bgr, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Exibir concentração e relaxamento na tela
            conc_text = f"Foco: {self.concentration:.2f}"
            relax_text = f"Relaxamento: {self.relaxation:.2f}"
            cv2.putText(frame_bgr, conc_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame_bgr, relax_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)
            
            # **Grava o frame no vídeo se a gravação estiver ativa**
            if self.running and self.video_writer:
                self.video_writer.write(frame_bgr)  # Escreve o frame no arquivo de vídeo

        self.root.after(33, self.update_camera)

    def fetch_openbci_data_thread(self):
        try:
            import requests
            response = requests.get("http://localhost:5000/data", timeout=1)
            if response.status_code == 200:
                data = response.json()
            else:
                data = {"status": "error", "message": f"Status code: {response.status_code}"}
        except Exception as e:
            data = {"status": "error", "message": str(e)}
        self.root.after(0, self.process_openbci_data, data)

    def process_openbci_data(self, data):
        if data.get("status") == "success":
            bands = data.get("data")
            if np.isscalar(bands) or not hasattr(bands, '__len__') or len(bands) < 5:
                self.text_output.insert(tk.END, "⚠️ Dados insuficientes no servidor\n")
            else:
                delta, theta, alpha, beta, gamma = bands[:5]
                timestamp = time.strftime('%H:%M:%S')

                self.text_output.insert(tk.END, f"\nTempo: {timestamp}\n")
                self.text_output.insert(tk.END, f"Delta: {delta:.5f} | Theta: {theta:.5f} | Alpha: {alpha:.5f} | Beta: {beta:.5f} | Gamma: {gamma:.5f}\n")
                self.text_output.see(tk.END)

                self.concentration = beta / (alpha + theta + delta + beta + gamma)
                self.relaxation = alpha / (alpha + theta + delta)

                self.update_focus_widget()
                
                # **Corrigido para escrever no CSV com apenas duas casas decimais**
                self.csv_writer.writerow([
                    timestamp,
                    f"{self.concentration:.2f}",  # Formata concentração para 2 casas decimais
                    f"{self.relaxation:.2f}",     # Formata relaxamento para 2 casas decimais
                    f"{delta:.5f}",
                    f"{theta:.5f}",
                    f"{alpha:.5f}",
                    f"{beta:.5f}",
                    f"{gamma:.5f}"
                ])

    def toggle_screen_recording(self):
        """Inicia ou para a gravação da tela"""
        if not hasattr(self, "recording") or not self.recording:
            self.recording = True
            self.screen_thread = threading.Thread(target=self.record_screen)
            self.screen_thread.start()
            self.record_btn.config(text="Parar Gravação")
        else:
            self.recording = False
            self.record_btn.config(text="Gravar Tela")
    
    def record_screen(self):
        """Grava a tela do computador e salva em um arquivo AVI"""
        fps = 10
        screen_size = tuple(pyautogui.size())
        codec = cv2.VideoWriter_fourcc(*"XVID")
        video = cv2.VideoWriter("tela_completa.avi", codec, fps, screen_size)

        while self.recording:
            frame = pyautogui.screenshot()
            frame = np.array(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
            time.sleep(1 / fps)

        video.release()
        print("Gravação de tela salva como 'screen_record.avi'")
    

    def update_focus_widget(self):
        self.focus_text.config(text=f"Foco: {self.concentration:.2f} \nRelaxamento: {self.relaxation:.2f}")

    def on_close(self):
        """Finaliza os processos e fecha a aplicação"""
        self.running = False
        if hasattr(self, "recording") and self.recording:
            self.recording = False  # Para a gravação da tela
        self.cap.release()
        self.csv_file.close()
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    threading.Thread(target=run_server, daemon=True).start()
    root = tk.Tk()
    app = OpenBCIWebcamApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
