import matplotlib.pyplot as plt   
  
# Enable LaTeX style  
plt.rc('text', usetex=True)  
plt.rc('font', family='serif') 
plt.rcParams.update({'font.size': 18})
  
# Data  
isso_70 = {  
    'localized change': [(0.51 + 0.03) / (1.75 + 2.82), (0.61 + 0.04) / (1.92 + 4.17), (0.88 + 0.08) / (2.04 + 5.8), (1.07 + 0.01) / (2.28 + 5.15)],
    'local scope': [(0 + 0.04) / (1.75 + 2.82), (0.11 + 0.06) / (1.92 + 4.17), (0.29 + 0.08) / (2.04 + 5.8), (0.51 + 0.0) / (2.28 + 5.15)],
    'consistency': [(1.44 + 2.76) / (1.75 + 2.82), (1.54 + 4.12) / (1.92 + 4.17), (1.76 + 5.62) / (2.04 + 5.8), (1.93 + 5.10) / (2.28 + 5.15)]
}  
isso_85 = {  
    'localized change': [(0.51 + 0.03) / (1.75 + 2.82), (0.46 + 0.0) / (1.69 + 3.97), (0.72 + 0.07) / (2.10 + 4.49), (0.69 + 0.14) / (2.11 + 4.74)],
    'local scope': [(0 + 0.04) / (1.75 + 2.82), (0.08 + 0.0) / (1.69 + 3.97), (0.30 + 0.11) / (2.10 + 4.49), (0.18 + 0.12) / (2.11 + 4.74)],
    'consistency': [(1.44 + 2.76) / (1.75 + 2.82), (1.56 + 3.94) / (1.69 + 3.97), (1.88 + 4.39) / (2.10 + 4.49), (1.92 + 4.64) / (2.11 + 4.74)]
}  
isso_100 = {  
    'localized change': [(0.51 + 0.03) / (1.75 + 2.82), (0.6 + 0.0) / (1.69 + 2.75), (0.6 + 0.0) / (1.68 + 3.47), (0.46 + 0.0) / (1.51 + 3.69)],
    'local scope': [(0 + 0.04) / (1.75 + 2.82), (0.04 + 0.0) / (1.69 + 2.75), (0.07 + 0.1) / (1.68 + 3.47), (0.06 + 0.0) / (1.51 + 3.69)],
    'consistency': [(1.44 + 2.76) / (1.75 + 2.82), (1.36 + 2.71) / (1.69 + 2.75), (1.35 + 3.43) / (1.68 + 3.47), (1.47 + 3.68) / (1.51 + 3.69)]
}  
constant_curve = {  
    'localized change': (0.89 + 0.24) / (2.46 + 5.81),
    'local scope': (0.61 + 0.15) / (2.46 + 5.81),
    'consistency': (2.36 + 5.80) / (2.46 + 5.81)
}  
  
# Plotting  
attributes = ['localized change', 'local scope', 'consistency']
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
    plt.ylabel('Fraction of Problem / Solution Feedback')  
    plt.xticks(range(4), labels=[0, 1, 2, 3])  
    plt.legend()  
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  
    plt.savefig('feedback_intrinsic_analysis/ps_evolution_{}.pdf'.format(attribute))
