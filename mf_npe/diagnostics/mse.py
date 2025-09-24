import os
from mf_npe.utils.calculate_error import ci95_t
import plotly.graph_objects as go

def plot_mse_barplot(df_mse, task_setup, net_init, true_xen, type_estimator, n_train_sims, ppc_path):
    # Compute means and 95% CI using t-distribution
    mse_mean, mse_ci = ci95_t(df_mse["mse"])
    mse_lf_mean, mse_lf_ci = ci95_t(df_mse["mse_lf"])
    mse_prior_mean, mse_prior_ci = ci95_t(df_mse["mse_prior"])

    # Save means to text file
    mean_file_name = f"mse_mean_{net_init}_{len(true_xen)}_{type_estimator}_{n_train_sims}_{task_setup.config_data['type_lf']}.txt"
    with open(os.path.join(ppc_path, mean_file_name), "w") as f:
        f.write(f"MSE: {mse_mean}\n")
        f.write(f"MSE LF: {mse_lf_mean}\n")
        f.write(f"MSE PRIOR: {mse_prior_mean}\n")

    # Plot setup
    bar_labels = ["MF-NPE", "NPE (LF)", "Prior"]
    bar_values = [mse_mean, mse_lf_mean, mse_prior_mean]
    error_bars = [mse_ci, mse_lf_ci, mse_prior_ci]
    bar_colors = ['#EA6340', '#234070', '#0F9E9A']
    bar_positions = [0, 1, 2]

    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=bar_positions,
        y=bar_values,
        error_y=dict(
            type='data',
            array=error_bars,
            visible=True,
            color='black',
            thickness=1.5,
            width=5,
        ),
        marker_color=bar_colors,
        textposition='auto'
    ))

    fig.update_layout(
        title=dict(
            text=f"Mean squared error over 1000 ppc, {len(true_xen)} x, LF: {task_setup.config_data['type_lf']}",
            font=dict(size=task_setup.title_size)
        ),
        font=dict(size=32),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title="",
            tickvals=bar_positions,
            ticktext=bar_labels,
            showticklabels=True
        ),
        yaxis=dict(
            title="MSE",
            range=[0, 5],
            gridcolor='lightgray',
            zeroline=True,
        )
    )

    fig.show()

    # Save plot
    fig.write_html(os.path.join(ppc_path, f"mse_plot_{task_setup.config_data['type_lf']}.html"))
    fig.write_image(os.path.join(ppc_path, f"mse_plot_{task_setup.config_data['type_lf']}.svg"))
    


    # def get_posterior_samples(self, true_xen, true_thetas, true_add_ons, posterior, lf_posterior, type_estimator, n_train_sims, net_init, true_posterior_samples=None) -> torch.Tensor:
    #     posterior_samples_over_x = []
    #     true_posterior_samples_over_x = []

    #     # The raw density estimator that does not have the reject-accept algorithm (needed when we have logit-transformed data)
    #     if isinstance(posterior, list) or type_estimator in ['active_npe'] or type_estimator in ['mf_tsnpe']:
    #         density_estimator = posterior
    #     else:
    #         density_estimator = posterior.__dict__['posterior_estimator']
            
    #     # mses = []
    #     # mses_lf = []
    #     # mses_prior = []
        
    #     for i, x in enumerate(tqdm(true_xen)):
    #         if self.task == 'task1':
    #             true_trace = true_add_ons['full_trace'][i]                            
    #         elif self.task == 'task2':
    #             true_trace = true_add_ons['full_trace'][i]
    #             I_curr = true_add_ons['inj_current']
    #             if i <= 3:
    #                 plot_CompNeuron_xen([x], I_curr, true_trace, self.config_data['dt'], "current true x", type_estimator, n_train_sims, i, self.main_path)
    #         elif self.task == 'task3':
    #             true_trace = true_add_ons['dataset'][i] # for meanfield: just rate
    #         else:
    #             raise ValueError("Task not recognized")


    #         # c2st is evaluated on approx. 30 samples (see task_setup)
    #         if self.evaluation_metric == 'c2st' or self.evaluation_metric == 'wasserstein' or self.evaluation_metric == 'mmd':
    #             if isinstance(density_estimator, list) and len(density_estimator) != 0:
    #                 # If density estimator is a list: For the sequential case: Then we have 
    #                 # A density estimator for each x, so we iterate the estimators over the number of x'en.
    #                 posterior_samples = density_estimator[i].sample((self.n_samples_to_generate,), x)
    #             else:
    #                 posterior_samples = density_estimator.sample((self.n_samples_to_generate,), x)
                
    #             posterior_samples_over_x.append(posterior_samples)    
                
    #             true_post_s = true_posterior_samples[i]
    #             true_posterior_samples_over_x.append(true_post_s)    
                
    #             # self._plot_pairplot_with_true_posterior(posterior_samples, true_post_s, true_thetas[i], 'estim vs true', n_train_sims)
                
    #             #self._plot_pairplot(true_post_s, true_thetas[i],'true posterior', 0)  
    #             #self._plot_pairplot(posterior_samples, true_thetas[i], type_estimator, n_train_sims)       
                
    #             # if lf_posterior is not None:
    #             #     lf_posterior_samples = lf_posterior.sample((self.n_samples_to_generate,), x) # Condition on the true x, and only for mf_npe
    #             #     # self._plot_pairplot_lf_hf(posterior_samples, lf_posterior_samples, true_thetas[i], type_estimator, n_train_sims)        
                    
    #             #     # Run posterior predictive checks for the first 30 xen for 1st task
    #             #     run_ppc = False
    #             #     if run_ppc:
    #             #         if i <= 30:
    #             #             x_pp, x_pp_lf, x_pp_prior = self._posterior_predictive_check_mf(posterior_samples, lf_posterior_samples, x, true_trace, type_estimator, n_train_sims, i)
                            
    #             #             # Compute per-sample squared error, then mean over samples
    #             #             mse = torch.mean(torch.mean((x_pp - x) ** 2, dim=1))        # shape: [1000] → scalar
    #             #             mse_lf = torch.mean(torch.mean((x_pp_lf - x) ** 2, dim=1))
    #             #             mse_prior = torch.mean(torch.mean((x_pp_prior - x) ** 2, dim=1))
    #             #             print(f"MSE for posterior predictive check (xen {i}): {mse.item()}")
    #             #             print(f"MSE LF for posterior predictive check (xen {i}): {mse_lf.item()}")
    #             #             print(f"MSE PRIOR for posterior predictive check (xen {i}): {mse_prior.item()}")
                            
    #             #             mses.append(mse)
    #             #             mses_lf.append(mse_lf)
    #             #             mses_prior.append(mse_prior)
                                          
    #         # nltp is evaluated on as much samples as possible (for paper: ~1000 samples)
    #         # Code is seperate because I do not want to sample from 1000 true xen.
    #         if self.evaluation_metric == 'nltp':
    #             # Plot only the first 3 pairplots: just to have an idea
    #             if i <= 3:
    #                 if isinstance(density_estimator, list):
    #                     # If density estimator is a list: For the sequential case: Then we have 
    #                     # A density estimator for each x, so we iterate the estimators over the number of x'en.
    #                     posterior_samples = posterior[i].sample((self.n_samples_to_generate,), x)
    #                 else:
    #                     posterior_samples = density_estimator.sample((self.n_samples_to_generate,), x)
                        
    #                 posterior_samples_over_x.append(posterior_samples)
                
    #                 # Make posterior samples that are filtered
    #                 # self._plot_pairplot(posterior_samples, true_thetas[i], type_estimator, n_train_sims)
    #                 # self._plot_conditional_pairplot(posterior, x)
                    
    #         # # Run posterior predictive checks for the first 30 xen for 1st task
    #         # if i <= 3:
    #         #     self._posterior_predictive_check(posterior_samples, lf_posterior_samples, x, true_trace, type_estimator, n_train_sims, i)
        
        
    #     # # Average over mses
    #     # mse_data = {
    #     #     "mse": [m.item() for m in mses],
    #     #     "mse_lf": [m.item() for m in mses_lf],
    #     #     "mse_prior": [m.item() for m in mses_prior],
    #     # }

    #     # # Convert to DataFrame
    #     # df_mse = pd.DataFrame(mse_data)
        
    #     # ppc_path = task_setup.main_path + "/ppc/"
    #     # os.makedirs(ppc_path, exist_ok=True)
    #     # pickle_name = f"mse_raw_{net_init}_{len(true_xen)}_{type_estimator}_{n_train_sims}_{task_setup.config_data['type_lf']}.pkl"
    #     # with open(os.path.join(ppc_path, pickle_name), "wb") as f:
    #     #     pickle.dump(df_mse, f)
            
    #     # plot_mse_barplot(df_mse, task_setup, net_init, true_xen, type_estimator, n_train_sims, ppc_path)

    #     posterior_samples = torch.stack(posterior_samples_over_x)

        
    #     save_dir = f"{self.main_path}/posterior_samples/"
    #     name = f"thetas_{net_init}_{len(true_xen)}_{type_estimator}_{n_train_sims}_{self.config_data['type_lf']}.p"
                
    #     post_samples = { 'posterior_samples': posterior_samples, 
    #                     'n_true_x': len(true_xen),
    #                     'true_theta': true_thetas,
    #                     'type_estimator': type_estimator,
    #                     'n_train_sims': n_train_sims}
        
    #     dump_pickle(save_dir, name, post_samples)
        
    #     if self.evaluation_metric == 'c2st' or self.evaluation_metric == 'wasserstein' or self.evaluation_metric == 'mmd':
    #         true_posterior_samples = torch.stack(true_posterior_samples_over_x)
    #         true_post_samples = { 'true_posterior_samples': true_posterior_samples}
    #         true_name = f"true_thetas_{len(true_xen)}.p"
            
    #         dump_pickle(save_dir, true_name, true_post_samples)
        
    #     return posterior_samples
