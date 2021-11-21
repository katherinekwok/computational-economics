
use mortgage_performance_data, clear
sum

local varlist  i_large_loan i_medium_loan rate_spread i_refinance score_0 age_r  cltv dti cu first_mort_r    i_FHA  i_open_year2-i_open_year5
tabstat  i_open_0 i_open_1 i_open_2 `varlist', stat(n mean sd min max) columns(statistics)

desc
forvalues i=0/2 {
	gen i_close_`i' = 1 - i_open_`i'
	probit i_close_`i'  score_`i' `varlist', r 
}
