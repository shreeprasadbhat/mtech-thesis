import numpy as np

BestFitParams = np.load('BestFitParameters.pickle', allow_pickle=True)

f = open('file.txt', 'w')

f.write('\\begin{table}\n')
f.write('\\centering\n')
f.write('\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n')
f.write('\\hline\n')
f.write('Correlation & sample & N & a & $a_err$ & b & $b_err$ & $\\sigma$ & $\\sigma_{int}$\\\\\n')
f.write('\\hline\n')

s = '\\multirow{3}{*}{$T_{lag}-L$} & low-z & 37 & ' + str(round(BestFitParams['T_lag-L']['low-z']['a'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['low-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['low-z']['b'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['low-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['low-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['low-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & high-z & 32 & ' + str(round(BestFitParams['T_lag-L']['high-z']['a'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['high-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['high-z']['b'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['high-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['high-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['high-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & All-z & 69 & ' + str(round(BestFitParams['T_lag-L']['All-z']['a'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['All-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['All-z']['b'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['All-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['All-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['T_lag-L']['All-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\hline\n')




s = '\\multirow{3}{*}{$V-L$} & low-z & 47 & ' + str(round(BestFitParams['V-L']['low-z']['a'],2)) + ' & ' + str(round(BestFitParams['V-L']['low-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['V-L']['low-z']['b'],2)) + ' & ' + str(round(BestFitParams['V-L']['low-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['V-L']['low-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['V-L']['low-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & high-z & 57 & ' + str(round(BestFitParams['V-L']['high-z']['a'],2)) + ' & ' + str(round(BestFitParams['V-L']['high-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['V-L']['high-z']['b'],2)) + ' & ' + str(round(BestFitParams['V-L']['high-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['V-L']['high-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['V-L']['high-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & All-z & 104 & ' + str(round(BestFitParams['V-L']['All-z']['a'],2)) + ' & ' + str(round(BestFitParams['V-L']['All-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['V-L']['All-z']['b'],2)) + ' & ' + str(round(BestFitParams['V-L']['All-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['V-L']['All-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['V-L']['All-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\hline\n')



s = '\\multirow{3}{*}{$E_{peak}-L$} & low-z & 50 & ' + str(round(BestFitParams['E_peak-L']['low-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['low-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['low-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['low-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['low-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['low-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & high-z & 66 & ' + str(round(BestFitParams['E_peak-L']['high-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['high-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['high-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['high-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['high-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['high-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & All-z & 116 & ' + str(round(BestFitParams['E_peak-L']['All-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['All-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['All-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['All-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['All-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-L']['All-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\hline\n')



s = '\\multirow{3}{*}{$E_{peak}-E_{\\gamma}$} & low-z & 12 & ' + str(round(BestFitParams['E_peak-E_gamma']['low-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['low-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['low-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['low-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['low-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['low-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & high-z & 12 & ' + str(round(BestFitParams['E_peak-E_gamma']['high-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['high-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['high-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['high-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['high-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['high-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & All-z & 24 & ' + str(round(BestFitParams['E_peak-E_gamma']['All-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['All-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['All-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['All-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['All-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_gamma']['All-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\hline\n')




s = '\\multirow{3}{*}{$T_{RT}-L$} & low-z & 39 & ' + str(round(BestFitParams['T_RT-L']['low-z']['a'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['low-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['low-z']['b'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['low-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['low-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['low-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & high-z & 40 & ' + str(round(BestFitParams['T_RT-L']['high-z']['a'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['high-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['high-z']['b'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['high-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['high-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['high-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & All-z & 79 & ' + str(round(BestFitParams['T_RT-L']['All-z']['a'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['All-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['All-z']['b'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['All-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['All-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['T_RT-L']['All-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\hline\n')




s = '\\multirow{3}{*}{$E_{peak}-E_{iso}$} & low-z & 40 & ' + str(round(BestFitParams['E_peak-E_iso']['low-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['low-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['low-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['low-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['low-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['low-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & high-z & 61 & ' + str(round(BestFitParams['E_peak-E_iso']['high-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['high-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['high-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['high-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['high-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['high-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\cline{2-9}\n')
s = ' & All-z & 101 & ' + str(round(BestFitParams['E_peak-E_iso']['All-z']['a'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['All-z']['a_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['All-z']['b'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['All-z']['b_err'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['All-z']['sigma_int'],2)) + ' & ' + str(round(BestFitParams['E_peak-E_iso']['All-z']['sigma_int_err'],2)) + '\\\\\n'
f.write(s)
f.write('\\hline\n')

f.write('\\end{tabular}\n')
f.write('\\caption{A test caption}\n')
f.write('\\label{table1}\n')
f.write('\\end{table}\n')

f.close()
