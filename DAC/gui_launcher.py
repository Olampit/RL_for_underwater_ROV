import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import os
from run_training import train as sac_train
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import sys

import matplotlib
matplotlib.use('Agg')

class RLGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ROV RL Training Launcher")
        self.pause_flag = threading.Event()
        


        self.vx_score_data = []
        self.vy_score_data = []
        self.vz_score_data = []
        self.roll_score_data = []
        self.pitch_score_data = []
        self.yaw_score_data = []

        
        
        self.vx_rate_data = []
        self.vy_rate_data = []
        self.vz_rate_data = []
        self.roll_rate_data = []
        self.pitch_rate_data = []
        self.yaw_rate_data = []
        
        self.critic_loss_data = []
        self.actor_loss_data = []
        self.mean_step_time_data = []
        self.q_value_data = []
        
        self.td_mean_data = []
        self.td_max_data = []
        self.td_min_data = []

        self.actor_grad_norm_data = []
        self.critic_grad_norm_data = []
        self.actor_weight_norm_data = []
        self.critic_weight_norm_data = []
        self.lr_data = []
        self.zeroes = []
        
        self.tracking_total_data = []
        self.stability_penalty_data = []
        self.reward_total_data = []

        self.fig2, (self.ax2, self.ax3) = plt.subplots(2, 1, figsize=(6, 2.5))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.canvas2.get_tk_widget().grid(row=8, column=1, sticky="nsew")

        self.fig3, self.ax4 = plt.subplots(figsize=(6, 2.5))
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=root)
        self.canvas3.get_tk_widget().grid(row=9, column=0, sticky="nsew")
        
        self.fig5, self.ax6 = plt.subplots(figsize=(6, 2.5))
        self.canvas5 = FigureCanvasTkAgg(self.fig5, master=root)
        self.canvas5.get_tk_widget().grid(row=9, column=1, columnspan=1, sticky="nsew")

        self.fig6, (self.ax7, self.ax10) = plt.subplots(2, 1, figsize=(6, 2.5)) 
        self.canvas6 = FigureCanvasTkAgg(self.fig6, master=root)
        self.canvas6.get_tk_widget().grid(row=10, column=0, columnspan=1, sticky="nsew")  
        
        self.fig7, (self.ax8, self.ax9) = plt.subplots(2, 1, figsize=(6, 2.5))  
        self.canvas7 = FigureCanvasTkAgg(self.fig7, master=root)
        self.canvas7.get_tk_widget().grid(row=10, column=1, sticky="nsew")
        
        self.fig8, self.ax11 = plt.subplots(figsize=(6, 2.5))
        self.canvas8 = FigureCanvasTkAgg(self.fig8, master=root)
        self.canvas8.get_tk_widget().grid(row=10, column=2, columnspan=2, sticky="nsew")


        self.agent_type = tk.StringVar(value="sac")
        ttk.Label(root, text="Agent Type:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(root, textvariable=self.agent_type, values=["sac"]).grid(row=0, column=1)

        self.episodes_var = tk.IntVar(value=40_000)
        self.max_steps_var = tk.IntVar(value=50)
        self.lr_var = tk.DoubleVar(value=3e-4)
        self.lr_var_end = tk.DoubleVar(value=1e-4)

        ttk.Label(root, text="Episodes:").grid(row=1, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.episodes_var).grid(row=1, column=1)

        ttk.Label(root, text="Max Steps:").grid(row=2, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.max_steps_var).grid(row=2, column=1)

        ttk.Label(root, text="Learning Rate (start):").grid(row=3, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.lr_var).grid(row=3, column=1)
        
        ttk.Label(root, text="Learning Rate (end):").grid(row=4, column=0, sticky="w")
        ttk.Entry(root, textvariable=self.lr_var_end).grid(row=4, column=1)


        self.train_button = ttk.Button(root, text="Start Training", command=self.launch_training)
        self.train_button.grid(row=6, column=0, pady=10)

        self.pause_button = ttk.Button(root, text="Pause", command=self.toggle_pause)
        self.pause_button.grid(row=6, column=1, pady=10)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=1000, mode="determinate")
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
        self.log_text.grid(row=7, column=0, columnspan=1, padx=(10, 5), pady=10, sticky="nsew")

        self.critic_loss_text = tk.Text(root, height=10, width=30, state="disabled", bg="#eaffea")
        self.critic_loss_text.grid(row=7, column=1, columnspan=1, padx=(5, 10), pady=10, sticky="nsew")


        self.critic_loss_text.tag_configure("normal", foreground="black")
        self.critic_loss_text.tag_configure("high", foreground="red", font=("TkDefaultFont", 9, "bold"))

        self.training_thread = None
        self.shutdown_flag = threading.Event()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def update_progress(self, current, total, reward, metrics=None):
        if self.shutdown_flag.is_set():
            return
        try:
            self.root.after(0, self._update_progress_main_thread, current, total, reward, metrics)
        except tk.TclError:
            pass  # La fenêtre a été fermée


    def _update_progress_main_thread(self, current, total, reward, metrics=None):
        if self.shutdown_flag.is_set() or not self.root.winfo_exists():
            return
        if self.pause_flag.is_set():
            return

        
        pct = int((current / total) * 100)
        try:
            self.progress["value"] = pct
        except tk.TclError:
            pass
    
        self.log(f"[EP {current}/{total}] Reward: {reward:.2f}")
        if metrics:
            self.update_critic_loss_text(metrics.get("critic_loss", 0.0))

        self.reward_values.append(reward)
        self.reward_line.set_data(range(1, len(self.reward_values) + 1), self.reward_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        if metrics:
            self.vx_rate_data.append(metrics.get("vx", 0.0))
            self.vy_rate_data.append(metrics.get("vy", 0.0))
            self.vz_rate_data.append(metrics.get("vz", 0.0))
            self.roll_rate_data.append(metrics.get("roll_rate", 0.0))
            self.pitch_rate_data.append(metrics.get("pitch_rate", 0.0))
            self.yaw_rate_data.append(metrics.get("yaw_rate", 0.0))


            self.vx_score_data.append(metrics.get("vx_score", 0.0))
            self.vy_score_data.append(metrics.get("vy_score", 0.0))
            self.vz_score_data.append(metrics.get("vz_score", 0.0))
            self.roll_score_data.append(metrics.get("roll_score", 0.0))
            self.pitch_score_data.append(metrics.get("pitch_score", 0.0))
            self.yaw_score_data.append(metrics.get("yaw_score", 0.0))


            
            
            self.critic_loss_data.append(metrics.get("critic_loss", 0.0))
            self.actor_loss_data.append(metrics.get("actor_loss", 0.0))

            self.mean_step_time_data.append(metrics.get("mean_step_time"))
            
            self.td_mean_data.append(metrics.get("td_mean", 0.0))
            self.td_max_data.append(metrics.get("td_max", 0.0))
            self.td_min_data.append(metrics.get("td_min", 0.0))
            self.actor_grad_norm_data.append(metrics.get("actor_grad_norm", 0.0))
            self.critic_grad_norm_data.append(metrics.get("critic_grad_norm", 0.0))
            self.actor_weight_norm_data.append(metrics.get("actor_weight_norm", 0.0))
            self.critic_weight_norm_data.append(metrics.get("critic_weight_norm", 0.0))
            self.lr_data.append(metrics.get("learning_rate", 0.0))

            self.zeroes.append(metrics.get("zero", 0.0))
            
            self.tracking_total_data.append(metrics.get("tracking_total", 0.0))
            self.stability_penalty_data.append(metrics.get("stability_penalty", 0.0))
            self.reward_total_data.append(metrics.get("reward_total", 0.0))


            all_metrics = [
                self.vx_rate_data, self.vy_rate_data, self.vz_rate_data,
                self.roll_rate_data, self.pitch_rate_data, self.yaw_rate_data,
                self.vx_score_data, self.vy_score_data, self.vz_score_data,
                self.roll_score_data, self.pitch_score_data, self.yaw_score_data,
                self.critic_loss_data, self.actor_loss_data,
                self.mean_step_time_data,
                self.td_mean_data, self.td_max_data, self.td_min_data,
                self.actor_grad_norm_data, self.critic_grad_norm_data,
                self.actor_weight_norm_data, self.critic_weight_norm_data,
                self.zeroes, self.stability_penalty_data, self.tracking_total_data, self.reward_total_data
            ]


            for data_list in all_metrics:
                if len(data_list) > 100:
                    data_list.pop(0)

            if len(self.lr_data) > 1000:
                self.lr_data.pop(0)
                    
                    
            self.ax2.cla()
            self.ax2.set_title("V_actual - V_goal")
            self.ax2.set_ylabel("m/s")
            self.ax2.plot(self.vx_rate_data, label="vx", color="blue")
            self.ax2.plot(self.vy_rate_data, label="vy", color="red")
            self.ax2.plot(self.vz_rate_data, label="vz", color="pink")
            self.ax2.plot(self.zeroes, label="0", color="black", linestyle="dashed")
            self.ax2.legend(loc = 'upper left')
            

            self.ax3.cla()
            self.ax3.set_ylabel("rad/s")
            self.ax3.set_xlabel("Episode")
            self.ax3.plot(self.roll_rate_data, label="roll", color="green")
            self.ax3.plot(self.pitch_rate_data, label="pitch", color="purple")
            self.ax3.plot(self.yaw_rate_data, label="yaw", color="orange")
            self.ax3.legend(loc = 'upper left')
            self.canvas2.draw()

            
                    
            self.ax4.cla()
            self.ax4.set_title("Reward Components")
            self.ax4.set_ylabel("Value")
            self.ax4.set_xlabel("Episode")
            self.ax4.plot(self.vx_score_data, label="vx", color="blue")
            self.ax4.plot(self.vy_score_data, label="vy", color="red")
            self.ax4.plot(self.vz_score_data, label="vz", color="pink")
            self.ax4.plot(self.roll_score_data, label="roll", color="green")
            self.ax4.plot(self.pitch_score_data, label="pitch", color="purple")
            self.ax4.plot(self.yaw_score_data, label="yaw", color="orange")
            self.ax4.legend(loc = 'upper left')
            self.canvas3.draw()
            
            

            self.ax6.cla()
            self.ax6.set_title("Training Diagnostics")
            self.ax6.set_ylabel("Value")
            self.ax6.set_xlabel("Episode")
            self.ax6.plot(self.critic_loss_data, label="Critic Loss", color="red")
            self.ax6.plot(self.actor_loss_data, label="Actor Loss", color="blue")
            self.ax6.legend(loc = 'upper left')
            self.canvas5.draw()
            
            
            
            self.ax7.cla()
            self.ax7.set_ylabel("s")
            self.ax7.plot(self.mean_step_time_data, label="Mean Step Time", color="orange")
            self.ax7.legend(loc='upper left')

            self.ax10.cla()
            self.ax10.set_ylabel("LR")
            self.ax10.set_xlabel("Episode")
            self.ax10.plot(self.lr_data, label="Actor LR", color="blue")
            self.ax10.legend(loc='upper left')

            self.canvas6.draw()


            
            self.ax8.cla()
            self.ax8.set_title("TD Error Stats")
            self.ax8.set_ylabel("TD Error")
            self.ax8.plot(self.td_mean_data, label="TD Mean", color="blue")
            self.ax8.plot(self.td_max_data, label="TD Max", color="red")
            self.ax8.plot(self.td_min_data, label="TD Min", color="green")
            self.ax8.legend(loc='upper left')

            self.ax9.cla()
            self.ax9.set_title("Gradient/Weight Norms")
            self.ax9.set_ylabel("Norm")
            self.ax9.set_xlabel("Episode")
            self.ax9.plot(self.actor_grad_norm_data, label="Actor Grad", color="purple")
            self.ax9.plot(self.critic_grad_norm_data, label="Critic Grad", color="orange")
            self.ax9.plot(self.actor_weight_norm_data, label="Actor W", linestyle="dashed", color="purple")
            self.ax9.plot(self.critic_weight_norm_data, label="Critic W", linestyle="dashed", color="orange")
            self.ax9.legend(loc='upper left')

            self.canvas7.draw()
            
            self.ax11.cla()
            self.ax11.set_title("Reward Breakdown")
            self.ax11.set_ylabel("Value")
            self.ax11.set_xlabel("Episode")
            self.ax11.plot(self.tracking_total_data, label="Tracking Total", color="blue")
            self.ax11.plot(self.stability_penalty_data, label="Stability Penalty", color="orange")
            self.ax11.plot(self.reward_total_data, label="Total Reward", color="green")
            self.ax11.axhline(y=0.0, color="gray", linestyle="dashed", linewidth=0.8)
            self.ax11.legend()
            self.canvas8.draw()







    def log(self, text):
        if not self.root.winfo_exists():
            return
        try:
            self.log_text.config(state="normal")
            self.log_text.insert("end", text + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        except tk.TclError:
            pass  # Ignore si la fenêtre est fermée


    def toggle_pause(self):
        if self.pause_flag.is_set():
            self.pause_flag.clear()
            self.pause_button.config(text="Pause")
        else:
            self.pause_flag.set()
            self.pause_button.config(text="Resume")


    def launch_training(self):
        
        self.vx_score_data.clear()
        self.vy_score_data.clear()
        self.vz_score_data.clear()
        self.roll_score_data.clear()
        self.pitch_score_data.clear()
        self.yaw_score_data.clear()

        
        
        self.vx_rate_data.clear()
        self.vy_rate_data.clear()
        self.vz_rate_data.clear()
        self.roll_rate_data.clear()
        self.pitch_rate_data.clear()
        self.yaw_rate_data.clear()
        
        self.critic_loss_data.clear()
        self.actor_loss_data.clear()
        self.mean_step_time_data.clear()
        self.q_value_data.clear()
        
        self.zeroes.clear()
        
        self.tracking_total_data.clear()
        self.stability_penalty_data.clear()
        self.reward_total_data.clear()


        self.ax.cla()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True)
        self.reward_line, = self.ax.plot([], [], label="Reward", color="blue")
        self.ax.legend(loc = 'upper left')
        self.canvas.draw()

        config = {
            "episodes": self.episodes_var.get(),
            "max_steps": self.max_steps_var.get(),
            "learning_rate_start": self.lr_var.get(),
            "learning_rate_end":self.lr_var_end.get(),
            "progress_callback": self.update_progress,
            "pause_flag": self.pause_flag,
            "shutdown_flag": self.shutdown_flag,
        
        }

        self.shutdown_flag.clear()
        self.training_thread = threading.Thread(target=self.run_training, args=(self.agent_type.get(), config))
        self.training_thread.start()


    def run_training(self, agent_type, config):
        try:
            if agent_type == "sac":
                sac_train(**config)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        except Exception as e:
            error_details = "".join(traceback.format_exception(*sys.exc_info()))
            self.log("Error occurred:\n" + error_details)
            messagebox.showerror("Training Error", f"An error occurred:\n\n{str(e)}\n\nCheck log for full traceback.")


    def on_closing(self):
        self.shutdown_flag.set()
        self.pause_flag.set()

        if self.training_thread and self.training_thread.is_alive():
            self.log("Stopping training thread...")
            self.training_thread.join(timeout=5.0)  # Give a bit more time
        
        # Instead of destroying here, just quit the mainloop
        self.root.quit()

        
    def notify_training_finished(self):
        # Called from training thread at end or shutdown to stop the GUI mainloop
        def stop_loop():
            if self.root.winfo_exists():
                self.log("Training finished or stopped. Closing GUI...")
                self.root.quit()
        self.root.after(0, stop_loop)
    
    def update_critic_loss_text(self, value):
        if not self.root.winfo_exists():
            return
        try:
            tag = "normal" if value <= 999.0 else "high"
            
            
            if value >= 1000:
                self.critic_loss_text.config(bg="#ffcccc")  # light red
            
            self.critic_loss_text.config(state="normal")
            self.critic_loss_text.insert("end", f"{value:.5f}\n", tag)
            self.critic_loss_text.see("end")
            self.critic_loss_text.config(state="disabled")
        except tk.TclError:
            pass




if __name__ == "__main__":
    root = tk.Tk()
    app = RLGui(root)
    root.mainloop()
    # After mainloop exits (e.g. after notify_training_finished called root.quit())
    if root.winfo_exists():
        root.destroy()



