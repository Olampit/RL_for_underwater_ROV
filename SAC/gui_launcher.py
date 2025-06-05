# gui_launcher.py
"""
Simple tkinter GUI to launch Q-learning or SAC training with adjustable parameters.
"""
import tkinter as tk
from tkinter import ttk
import threading
from tkinter import messagebox

# --- Real SAC import ---
from run_training import train as sac_train

# --- Placeholder for Q-learning (optional) ---
# from q_learning.run_training import train as q_train


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


class RLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ROV RL Training Launcher")

        # Agent selection
        self.agent_type = tk.StringVar(value="sac")
        ttk.Label(root, text="Agent Type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(root, textvariable=self.agent_type, values=["sac", "q_learning"]).grid(row=0, column=1)

        # Parameters
        self.episodes_var = tk.IntVar(value=200)
        self.max_steps_var = tk.IntVar(value=300)
        self.lr_var = tk.DoubleVar(value=3e-4)

        ttk.Label(root, text="Episodes:").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.episodes_var).grid(row=1, column=1)

        ttk.Label(root, text="Max Steps:").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.max_steps_var).grid(row=2, column=1)

        ttk.Label(root, text="Learning Rate:").grid(row=3, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.lr_var).grid(row=3, column=1)

        # Progress bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=10)

        # Launch button
        self.train_button = ttk.Button(root, text="Lancer l'entraînement", command=self.launch_training)
        self.train_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Log output
        self.log_text = tk.Text(root, height=10, width=50, state="disabled")
        self.log_text.grid(row=6, column=0, columnspan=2, pady=10)

    def update_progress(self, current, total, reward):
        pct = int((current / total) * 100)
        self.progress["value"] = pct
        self.log(f"[EP {current}/{total}] Reward: {reward:.2f}")

    def log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def launch_training(self):
        config = {
            "episodes": self.episodes_var.get(),
            "max_steps": self.max_steps_var.get(),
            "learning_rate": self.lr_var.get(),
            "progress_callback": self.update_progress
        }
        start_training_from_gui(self.agent_type.get(), config)


if __name__ == "__main__":
    root = tk.Tk()
    app = RLGui(root)
    root.mainloop()
