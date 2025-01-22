import matplotlib.pyplot as plt   
  
# Enable LaTeX style  
plt.rc('text', usetex=True)  
plt.rc('font', family='serif') 
plt.rcParams.update({'font.size': 18})
  
# Data  
isso_70 = {  
    'praise': [(2.15 + 2.05 + 2.15) / 3, 1.89, 1.86, 1.83],  
    'problem': [(1.99 + 1.90 + 1.99) / 3, 2.15, 2.04, 2.28],  
    'solution': [(3.40 + 3.33 + 3.36) / 3, 4.51, 5.80, 5.15]  
}  
isso_85 = {  
    'praise': [(2.15 + 2.05 + 2.15) / 3, 1.99, 1.79, 1.70],  
    'problem': [(1.99 + 1.90 + 1.99) / 3, 1.85, 2.14, 2.11],  
    'solution': [(3.40 + 3.33 + 3.36) / 3, 4.29, 4.54, 4.75]  
}  
isso_100 = {  
    'praise': [(2.15 + 2.05 + 2.15) / 3, 2.24, 2.22, 1.96],  
    'problem': [(1.99 + 1.90 + 1.99) / 3, 1.90, 1.85, 1.62],  
    'solution': [(3.40 + 3.33 + 3.36) / 3, 3.33, 3.75, 3.91]  
}  
constant_curve = {  
    'praise': 2.79,  
    'problem': 2.46,  
    'solution': 5.80  
}  
  
# Plotting  
attributes = ['praise', 'problem', 'solution']  
models = ['\\textsc{PROF}, $\\tau=0.7$', '\\textsc{PROF}, $\\tau=0.85$', '\\textsc{PROF}, $\\tau=1.0$', '\\texttt{gpt-4}']  
colors = ['#D81B60', '#1E88E5', '#FFC107', '#004D40']
model_key_list = ['isso_70', 'isso_85', 'isso_100', 'constant_curve']
markers = ['v', 'o', '^', 's']
  
for i, attribute in enumerate(attributes):  
    plt.figure(figsize=(8, 6))  
      
    for j, model in enumerate(models):  
        model_key = model_key_list[j]  
        if model_key != 'constant_curve':  
            plt.plot(range(len(eval(model_key)[attribute])), eval(model_key)[attribute], marker=markers[j], label=model, color=colors[j])  
        else:  
            plt.axhline(y=eval(model_key)[attribute], linestyle='--', label=model, color=colors[j])  
      
    plt.title(attribute.capitalize())  
    plt.xlabel('Iteration')  
    plt.ylabel('Average Number per Feedback')  
    plt.xticks(range(4), labels=[0, 1, 2, 3])  
    plt.legend()  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  
    plt.savefig('feedback_intrinsic_analysis/evolution_{}.pdf'.format(attribute))
