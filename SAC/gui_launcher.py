import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from run_training import train as sac_train
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ROV RL Training Launcher")

        self.pause_flag = threading.Event()

        self.vx_actual_data = []
        self.vx_target_data = []
        self.yaw_data = []
        self.pitch_rate_data = []
        self.roll_rate_data = []
        self.velocity_data = []
        self.std_data = []
        self.bonus_data = []
        self.critic_loss_data = []
        self.actor_loss_data = []
        self.entropy_data = []
        self.mean_step_time_data = []
        self.q_value_data = []

        self.fig2, (self.ax2, self.ax3) = plt.subplots(2, 1, figsize=(6, 2.5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.canvas2.get_tk_widget().grid(row=8, column=1, sticky="nsew")

        self.fig3, self.ax4 = plt.subplots(figsize=(6, 2.5))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=root)
        self.canvas3.get_tk_widget().grid(row=9, column=0, sticky="nsew")
        
        self.fig5, self.ax6 = plt.subplots(figsize=(6, 2.5))
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=root)
        self.canvas5.get_tk_widget().grid(row=9, column=1, columnspan=2, sticky="nsew")

        self.fig6, self.ax7 = plt.subplots(figsize=(6, 2.5))
        self.canvas6 = FigureCanvasTkAgg(self.fig6, master=root)
        self.canvas6.get_tk_widget().grid(row=10, column=0, columnspan=3, sticky="nsew")
        
        self.fig7, self.ax8 = plt.subplots(figsize=(6, 2.5))
        self.canvas8 = FigureCanvasTkAgg(self.fig6, master=root)
        self.canvas8.get_tk_widget().grid(row=10, column=0, columnspan=3, sticky="nsew")

        self.agent_type = tk.StringVar(value="sac")
        ttk.Label(root, text="Agent Type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(root, textvariable=self.agent_type, values=["sac"]).grid(row=0, column=1)

        self.episodes_var = tk.IntVar(value=200000)
        self.max_steps_var = tk.IntVar(value=50)
        self.lr_var = tk.DoubleVar(value=3e-3)

        ttk.Label(root, text="Episodes:").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.episodes_var).grid(row=1, column=1)

        ttk.Label(root, text="Max Steps:").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.max_steps_var).grid(row=2, column=1)

        ttk.Label(root, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.lr_var).grid(row=3, column=1)

        self.train_button = ttk.Button(root, text="Start Training", command=self.launch_training)
        self.train_button.grid(row=6, column=0, pady=10)

        self.pause_button = ttk.Button(root, text="Pause", command=self.toggle_pause)
        self.pause_button.grid(row=6, column=1, pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=3, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 2.5))
        self.reward_values = []
        self.reward_line, = self.ax.plot([], [], label="Reward", color="blue")
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=8, column=0, sticky="nsew")

        self.log_text = tk.Text(root, height=10, width=50, state="disabled")
        self.log_text.grid(row=7, column=0, columnspan=3, pady=10)

    def update_progress(self, current, total, reward, metrics=None):
        if self.pause_flag.is_set():
            return

        pct = int((current / total) * 100)
        self.progress["value"] = pct
        self.log(f"[EP {current}/{total}] Reward: {reward:.2f}")

        self.reward_values.append(reward)
        self.reward_line.set_data(range(1, len(self.reward_values) + 1), self.reward_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        if metrics:
            self.vx_actual_data.append(metrics.get("vx", 0.0))
            self.vx_target_data.append(metrics.get("vx_target", 0.3))

            self.ax2.cla()
            self.ax2.set_title("vx: Velocity Tracking")
            self.ax2.set_ylabel("vx (m/s)")
            self.ax2.plot(self.vx_actual_data, label="vx_actual", color="blue")
            self.ax2.plot(self.vx_target_data, label="vx_target", color="cyan", linestyle="dashed")
            self.ax2.legend()

            self.yaw_data.append(metrics.get("yaw_rate", 0.0))
            self.pitch_rate_data.append(metrics.get("pitch_rate", 0.0))
            self.roll_rate_data.append(metrics.get("roll_rate", 0.0))

            self.ax3.cla()
            self.ax3.set_title("Angular Rates")
            self.ax3.set_ylabel("rad/s")
            self.ax3.set_xlabel("Episode")
            self.ax3.plot(self.yaw_data, label="yaw", color="orange")
            self.ax3.plot(self.pitch_rate_data, label="pitch", color="purple")
            self.ax3.plot(self.roll_rate_data, label="roll", color="green")
            self.ax3.legend()
            self.canvas2.draw()

            self.velocity_data.append(metrics.get("velocity_score", 0.0))
            self.std_data.append(metrics.get("std_score", 0.0))
            self.bonus_data.append(metrics.get("bonus", 0.0))

            self.ax4.cla()
            self.ax4.set_title("Reward Components")
            self.ax4.set_ylabel("Value")
            self.ax4.set_xlabel("Episode")
            self.ax4.plot(self.velocity_data, label="velocity_score", color="blue")
            self.ax4.plot(self.std_data, label="std_score", color="gray")
            self.ax4.plot(self.bonus_data, label="bonus", color="magenta")
            self.ax4.legend()
            self.canvas3.draw()
            
            self.critic_loss_data.append(metrics.get("critic_loss", 0.0))
            self.actor_loss_data.append(metrics.get("actor_loss", 0.0))
            self.entropy_data.append(metrics.get("entropy", 0.0))

            self.ax6.cla()
            self.ax6.set_title("Training Diagnostics")
            self.ax6.set_ylabel("Value")
            self.ax6.set_xlabel("Episode")
            self.ax6.plot(self.critic_loss_data, label="Critic Loss", color="red")
            self.ax6.plot(self.actor_loss_data, label="Actor Loss", color="blue")
            self.ax6.plot(self.entropy_data, label="Entropy", color="green")
            self.ax6.legend()
            self.canvas5.draw()
            
            
            self.mean_step_time_data.append(metrics.get("mean_step_time"))
            self.ax7.cla()
            self.ax7.set_title("Mean Step Time")
            self.ax7.set_ylabel("Time")
            self.ax7.set_xlabel("Episode")
            self.ax7.plot(self.mean_step_time_data, label="Mean Step Time", color="orange")
            self.ax7.legend()
            self.canvas6.draw()
            
            
            self.q_value_data.append(metrics.get("mean_q_value"))
            self.ax8.cla()
            self.ax8.set_title("Mean Q-Value")
            self.ax8.set_ylabel("Q(s, a)")
            self.ax8.set_xlabel("Episode")
            self.ax8.plot(self.q_value_data, label="Q-Value", color="blue")
            self.ax8.legend()
            self.canvas8.draw()




    def log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def toggle_pause(self):
        if self.pause_flag.is_set():
            self.pause_flag.clear()
            self.pause_button.config(text="Pause")
        else:
            self.pause_flag.set()
            self.pause_button.config(text="Resume")


    def launch_training(self):
        self.reward_values.clear()
        self.vx_actual_data.clear()
        self.vx_target_data.clear()
        self.yaw_data.clear()
        self.pitch_rate_data.clear()
        self.roll_rate_data.clear()
        self.velocity_data.clear()
        self.std_data.clear()
        self.bonus_data.clear()
        self.q_value_data.clear()

        self.ax.cla()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.reward_line, = self.ax.plot([], [], label="Reward", color="blue")
        self.ax.legend()
        self.canvas.draw()

        config = {
            "episodes": self.episodes_var.get(),
            "max_steps": self.max_steps_var.get(),
            "learning_rate": self.lr_var.get(),
            "progress_callback": self.update_progress,
            "pause_flag": self.pause_flag,
        
        }

        threading.Thread(target=self.run_training, args=(self.agent_type.get(), config), daemon=True).start()

    def run_training(self, agent_type, config):
        try:
            if agent_type == "sac":
                sac_train(**config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = RLGui(root)
    root.mainloop()
