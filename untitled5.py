# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:43:46 2020

@author: 37212
"""
        
        
# =============================================================================
#         data=Data()
#         fund_manager_custodianfeeratio=data.fund_manager_custodianfeeratio
#         fund_manager_custodianfeeratio.reset_index(inplace=True)
#         fund_manager_custodianfeeratio.set_index(['manager_ID','firstinvesttype'],inplace=True)
#         
# =============================================================================
        fund_manager_custodianfeeratio = CALFUNC.del_dat_early_than(fund_manager_custodianfeeratio, START_YEAR)
        fund_manager_custodianfeeratio=tool3.cleaning(fund_manager_custodianfeeratio)
        
        
        
        
        
        
# =============================================================================
#         data=Data()
#         fund_manager_managementfeeratio=data.fund_manager_managementfeeratio
#         fund_manager_managementfeeratio.reset_index(inplace=True)
#         fund_manager_managementfeeratio.set_index(['manager_ID','firstinvesttype'],inplace=True)
# =============================================================================
        fund_manager_managementfeeratio = CALFUNC.del_dat_early_than(fund_manager_managementfeeratio, START_YEAR)
        fund_manager_managementfeeratio=tool3.cleaning(fund_manager_managementfeeratio)
        
        



        
# =============================================================================
#         data=Data()
#         fund_manager_purchasefeeratio=data.fund_manager_purchasefeeratio
#         fund_manager_purchasefeeratio.reset_index(inplace=True)
#         fund_manager_purchasefeeratio.set_index(['manager_ID','firstinvesttype'],inplace=True)
# =============================================================================
        fund_manager_purchasefeeratio = CALFUNC.del_dat_early_than(fund_manager_purchasefeeratio, START_YEAR)
        fund_manager_purchasefeeratio=tool3.cleaning(fund_manager_purchasefeeratio)
        