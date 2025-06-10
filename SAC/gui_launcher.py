import tkinter as tk
from tkinter import ttk, messagebox
import threading
import os

# --- Real SAC import ---
from run_training import train as sac_train

# --- Matplotlib for reward plot ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ROV RL Training Launcher")

        # --- Extra plots: vx and yaw_rate ---
        self.vx_actual_data = []
        self.vx_target_data = []
        self.yaw_data = []
        self.yaw_target_data = []

        self.fig2, (self.ax2, self.ax3) = plt.subplots(2, 1, figsize=(5, 3))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.canvas2.get_tk_widget().grid(row=9, column=0, columnspan=2)

        # --- Reward component plot ---
        self.forward_data = []
        self.sideways_data = []
        self.upward_data = []
        self.stability_data = []
        self.bonus_data = []

        self.fig3, self.ax4 = plt.subplots(figsize=(5, 3))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=root)
        self.canvas3.get_tk_widget().grid(row=10, column=0, columnspan=2)

        self.agent_type = tk.StringVar(value="sac")
        ttk.Label(root, text="Agent Type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(root, textvariable=self.agent_type, values=["sac", "q_learning"]).grid(row=0, column=1)

        self.episodes_var = tk.IntVar(value=5000)
        self.max_steps_var = tk.IntVar(value=300)
        self.lr_var = tk.DoubleVar(value=1e-4)

        ttk.Label(root, text="Episodes:").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.episodes_var).grid(row=1, column=1)

        ttk.Label(root, text="Max Steps:").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.max_steps_var).grid(row=2, column=1)

        ttk.Label(root, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.lr_var).grid(row=3, column=1)

        self.resume_var = tk.BooleanVar(value=False)
        resume_check = ttk.Checkbutton(root, text="Resume from checkpoint", variable=self.resume_var)
        resume_check.grid(row=4, column=0, columnspan=2, sticky="w")

        if not os.path.exists("sac_checkpoint.pt"):
            self.resume_var.set(False)
            resume_check.config(state="disabled")

        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=5, column=0, columnspan=2, pady=10)

        self.train_button = ttk.Button(root, text="Lancer l'entraînement", command=self.launch_training)
        self.train_button.grid(row=6, column=0, columnspan=2, pady=10)

        self.log_text = tk.Text(root, height=10, width=50, state="disabled")
        self.log_text.grid(row=7, column=0, columnspan=2, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.reward_values = []
        self.reward_line, = self.ax.plot([], [], label="Reward", color="blue")
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=2)

    def update_progress(self, current, total, reward, metrics=None):
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
            self.vx_target_data.append(metrics.get("vx_target", 0.0))
            self.yaw_data.append(metrics.get("yaw_rate", 0.0))
            self.yaw_target_data.append(metrics.get("yaw_target", 0.0))

            self.ax2.cla()
            self.ax2.set_title("Velocity vs. Target")
            self.ax2.set_ylabel("vx (m/s)")
            self.ax2.plot(self.vx_actual_data, label="vx_actual", color="blue")
            self.ax2.plot(self.vx_target_data, label="vx_target", color="cyan")
            self.ax2.legend()

            self.ax3.cla()
            self.ax3.set_ylabel("yaw rate (rad/s)")
            self.ax3.set_xlabel("Episode")
            self.ax3.plot(self.yaw_data, label="yaw_rate", color="orange")
            self.ax3.plot(self.yaw_target_data, label="yaw_target", color="red")
            self.ax3.legend()

            self.canvas2.draw()

            self.forward_data.append(metrics.get("forward", 0.0))
            self.sideways_data.append(metrics.get("sideways", 0.0))
            self.upward_data.append(metrics.get("upward", 0.0))
            self.stability_data.append(metrics.get("stability", 0.0))
            self.bonus_data.append(metrics.get("bonus", 0.0))

            self.ax4.cla()
            self.ax4.set_title("Reward Components")
            self.ax4.set_ylabel("Reward Value")
            self.ax4.set_xlabel("Episode")
            self.ax4.plot(self.forward_data, label="forward", color="green")
            self.ax4.plot(self.sideways_data, label="sideways", color="blue")
            self.ax4.plot(self.upward_data, label="upward", color="cyan")
            self.ax4.plot(self.stability_data, label="stability", color="orange")
            self.ax4.plot(self.bonus_data, label="bonus", color="magenta")
            self.ax4.legend()
            self.canvas3.draw()

    def log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def launch_training(self):
        self.reward_values.clear()
        self.vx_actual_data.clear()
        self.vx_target_data.clear()
        self.yaw_data.clear()
        self.yaw_target_data.clear()
        self.forward_data.clear()
        self.sideways_data.clear()
        self.upward_data.clear()
        self.stability_data.clear()
        self.bonus_data.clear()

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
            "resume": self.resume_var.get()
        }

        start_training_from_gui(self.agent_type.get(), config)

def start_training_from_gui(agent_type, config):
    def run():
        try:
            if agent_type == "sac":
                sac_train(**config)
            elif agent_type == "q_learning":
                print("[ERROR] Q-learning integration not yet implemented.")
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))

    threading.Thread(target=run, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = RLGui(root)
    root.mainloop()
