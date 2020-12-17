# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:04:20 2020

@author: 37212
"""
data=Data()
fund_manager_alpha=data.fund_manager_alpha
fund_manager_beta=data.fund_manager_beta
fund_manager_r2=data.fund_manager_r2
fund_manager_alpha=fund_manager_alpha.T
fund_manager_beta=fund_manager_beta.T
fund_manager_r2=fund_manager_r2.T
fund_manager_alpha.to_excel(os.path.join(basic_path, 'fund_manager_alpha.xlsx'))
fund_manager_beta.to_excel(os.path.join(basic_path, 'fund_manager_beta.xlsx'))
fund_manager_r2.to_excel(os.path.join(basic_path, 'fund_manager_r2.xlsx'))