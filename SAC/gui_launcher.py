import tkinter as tk
from tkinter import ttk, messagebox
import threading

# --- Real SAC import ---
from run_training import train as sac_train

# --- Matplotlib for reward plot ---
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ROV RL Training Launcher")

        self.agent_type = tk.StringVar(value="sac")
        ttk.Label(root, text="Agent Type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(root, textvariable=self.agent_type, values=["sac", "q_learning"]).grid(row=0, column=1)

        self.episodes_var = tk.IntVar(value=5000)
        self.max_steps_var = tk.IntVar(value=1000)
        self.lr_var = tk.DoubleVar(value=1e-4)

        ttk.Label(root, text="Episodes:").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.episodes_var).grid(row=1, column=1)

        ttk.Label(root, text="Max Steps:").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.max_steps_var).grid(row=2, column=1)

        ttk.Label(root, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.lr_var).grid(row=3, column=1)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=10)

        self.train_button = ttk.Button(root, text="Lancer l'entraînement", command=self.launch_training)
        self.train_button.grid(row=5, column=0, columnspan=2, pady=10)

        self.log_text = tk.Text(root, height=10, width=50, state="disabled")
        self.log_text.grid(row=6, column=0, columnspan=2, pady=10)

        # --- Matplotlib Figure for Reward Plot ---
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.reward_values = []
        self.reward_line, = self.ax.plot([], [], label="Reward", color="blue")
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=2)

    def update_progress(self, current, total, reward):
        pct = int((current / total) * 100)
        self.progress["value"] = pct
        self.log(f"[EP {current}/{total}] Reward: {reward:.2f}")

        # --- Update plot ---
        self.reward_values.append(reward)
        self.reward_line.set_data(range(1, len(self.reward_values) + 1), self.reward_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def launch_training(self):
        self.reward_values.clear()  # Reset plot
        self.ax.cla()  # Clear axis
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
            "progress_callback": self.update_progress
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
