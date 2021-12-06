# Author: Katherine Kwok
# Date: December 5, 2021

# This file contains the code for Problem Set 3 (for JF's portion) a.k.a.
# Problem Set 10 overall. The main program implements the BLP algorithm.

# ---------------------------------------------------------------------------- #
#   (0) declare primitives and functions for reading data
# ---------------------------------------------------------------------------- #

# Primitives: This struct stores all the parameter values for the BLP algo.
@with_kw struct Primitives

    car_char_varlist = ["price","dpm","hp2wt","size","turbo","trans","Year_1986",
						"Year_1987","Year_1988","Year_1989","Year_1990","Year_1991",
						"Year_1992","Year_1993","Year_1994","Year_1995","Year_1996",
						"Year_1997","Year_1998","Year_1999","Year_2000","Year_2001",
						"Year_2002","Year_2003","Year_2004","Year_2005","Year_2006",
						"Year_2007","Year_2008","Year_2009","Year_2010","Year_2011",
						"Year_2012","Year_2013","Year_2014","Year_2015","model_class_2",
						"model_class_3","model_class_4","model_class_5","cyl_2",
						"cyl_4","cyl_6","cyl_8","drive_2","drive_3","Intercept"]

	car_exog_varlist = car_char_varlist[2:end]

	car_iv_varlist = ["i_import","diffiv_local_0","diffiv_local_1","diffiv_local_2",
					  "diffiv_local_3","diffiv_ed_0"]

	outcome_varlist = ["share", "delta_iia", "price"]
end

# DataSets: This struct stores all the data sets and related information that are
#		    used in the BLP algorithm.
mutable struct DataSets

	car_pid::Array{Any}	# array of car pids for each market/year

	X::Array{Any}		# car characteristics
	IV::Array{Any}		# IV variables including exog. car characteristic and instruments
	Z::Array{Any}		# non-linear attribute (price)

	mkt_share::Array{Any}	# outcome: market share
	delta_iia::Array{Any}	# outcome: quality index
	delta_0::Array{Any}
	car_price::Array{Any}	# outcome: price

	eta::Array{Any}			# simulated type (eta)
	eta_Z::Array{Any}		# interaction between simulated type and Z
	n_eta::Int64			# number of simulated types

end

# print_stats_func: This function gets the means for a given data frame and prints
#					out the means line by line.
function print_stats_func(dataframe, varlist, title)
	means = round.(mean.(eachcol(dataframe)), digits = 3) # get mean for each variable
	stats = hcat(varlist, means)						  # merge with var names
	output_title = " " * title * ": "

	@printf "+--------------------------------------------------------------+\n"
	println(output_title)
	@printf "+--------------------------------------------------------------+\n"
	# print each mean with variable name
	for ind in 1:size(stats, 1)
		@printf " mean %s = %.3f \n" stats[ind, 1] stats[ind, 2]
	end
	@printf "+--------------------------------------------------------------+\n"
end

# read_car_data: This function reads and processes the car characteristics data set
#		         and car instruments data set to get the car attribute variables,
#				 outcome variables, and instrument variables.
#				 This function also stores the unique product IDs for each market.

function read_car_data(prim, car_char_path, car_iv_path; print_stats = true)
	@unpack car_char_varlist, car_iv_varlist, car_exog_varlist, outcome_varlist = prim

	dt1 = DataFrame(load(car_char_path)) # load in car characteristic data set
	dt2 = DataFrame(load(car_iv_path)) 	 # load in car instruments data set

	# get outcome variables from car characteristic data: market shares, delta, price
	mkt_share = Array(select(dt1, "share"))
	delta_iia = Array(select(dt1, "delta_iia"))
	delta_0 = delta_iia
	car_price = Array(select(dt1, "price"))

	X = select(dt1, car_char_varlist)   # select car characteristics variables
	iv = select(dt2, car_iv_varlist)    # select instrument variables
	Z = select(X, "price") 				# get non-linear attribute

	# get IV variables (merge exogenous car characteristics with instruments)
	IV = hcat(select(X, car_exog_varlist), iv)

	# if we want to output means of the characteristic variables
	if print_stats == true
		print_stats_func(X, car_char_varlist, "Average car characteristics")
		print_stats_func(iv, car_iv_varlist, "Average car instruments: ")
	end

	# generate product IDs: treating each year as a different market, get the row indices
	# for each market and store as product IDs
	year_data = Array(select(dt1, "Year"))
	year_list = sort(unique(year_data))
	pid = fill([], size(year_list, 1))

	for t in 1:size(year_list, 1) # store IDs within each market
		pid[t] = findall(x -> x == year_list[t], year_data)
	end
	pid = hcat(year_list, pid) # store the year/market with corresponding product IDs

	return mkt_share, delta_iia, delta_0, car_price, Array(X), Array(Z), Array(IV), pid
end


# read_type_data: This function reads and summarizes the simulated type data (income).
#			      It returns the types (eta) and the interaction between Z (price)
#				  and the random types (eta).
function read_type_data(type_data_path, Z, pid)

	eta = Array(DataFrame(load(type_data_path))) # load in type data set
	eta_Z = fill(Array{Any, 2}(undef, 0, 0), size(pid, 1))

	# pre-compute interaction between Z (price) and types (eta)
	for t in 1:size(pid, 1)
		eta_Z[t] = Z[CartesianIndex.(pid[t, 2])] * eta'
	end

	return eta, eta_Z
end


# process_data: This function reads the data sets and prepares them for the
#				BLP algorithm as a DataSets structure.

function process_data(prim, car_char_path, car_iv_path, type_data_path)

	@printf "+--------------------------------------------------------------+\n"
	@printf "  Processing car and simulated type data... \n"
	@printf "+--------------------------------------------------------------+\n"

	# read in car characteristic, instruments, and outcome data
	mkt_share, delta_iia, delta_0, car_price, X, Z, IV, pid = read_car_data(prim, car_char_path, car_iv_path)

	# read in simulated type (income) data
	eta, eta_Z = read_type_data(type_data_path, Z, pid)
	n_eta = size(eta, 1)

	# initialize DataSets struct
	dataset = DataSets(pid, X, IV, Z, mkt_share, delta_iia, delta_0, car_price, eta, eta_Z, n_eta)

	return dataset
end


# ---------------------------------------------------------------------------- #
#   (1) inverting demand functions
# ---------------------------------------------------------------------------- #

# value: This function computes utility (the idiosyncratic part) within a given
#	     market for each product and consumer type pairing.
function value(λ_p, eta_Z_t)

	μ = zeros(size(eta_Z_t))
	for i in 1:size(λ_p, 1)
		μ .+= λ_p[i] * eta_Z_t[i]	# formula for idiosyncratic part of utility
	end
	return exp.(μ)		# return exponentiated utility
end

# demand: This function computes the demand
function demand(pid_t, μ, vdelta, n_eta; get_jacob = 0)

	exp_V = exp.(vdelta[pid_t]) .* μ           # multiply exp delta for mkt with idiosyncratic utility
	σ = exp_V ./ (1 .+ sum(exp_V, dims = 1))   # formula for sigma (denominator does column sum of exp_V)
	σ_hat = mean(σ, dims = 2)				   # get row means of sigma

	if get_jacob == 1		# get jacobian matrix if requested
		m = (σ * σ')/n_eta
		m[diagind(m)] .= 0
		jacob = Diagonal(mean((σ .* (1 .- σ)), dims = 2)) .- m
	else
		jacob = [0]
	end

	return σ_hat, jacob

end

# inverse: This function inverts the demand function
function inverse(dataset, λ_p, ε_1, tol, output_path, method; max_iter = 1000, max_T = 1)

	@unpack car_pid, delta_0, mkt_share, eta_Z, n_eta = dataset

	vdelta = delta_0						# initialize parameter delta
	norm_mat = Array{Any, 2}(undef, 0, 2)	# matrix to store norms
	T = size(car_pid, 1)					# number of markets
	iter = zeros(T, 1)						# iteration counter

	for t in 1:max_T						# for each market

		pid_t = CartesianIndex.(car_pid[t, 2])	# get product IDs for given market
		μ = value(λ_p, eta_Z[t])                # pre-compute utility that is independent of delta
		f = fill(1000, 1, 1)  					# initialize f value
		norm_f = norm(f)							# initialize norm value

		while norm_f > tol && iter[t] < max_iter

			if (norm_f > ε_1) # if norm is larger than 1, evaluate demand without jacobian
				σ_hat, jacob = demand(pid_t, μ, vdelta, n_eta; get_jacob = 0)
				f = log.(mkt_share[pid_t]) .- log.(σ_hat)
				vdelta[pid_t] = vdelta[pid_t] + f # contraction mapping step

			else			   # if norm is smaller than 1, evaluate demand with jacobian
				σ_hat, jacob = demand(pid_t, μ, vdelta, n_eta; get_jacob = 1)
				f = log.(mkt_share[pid_t]) .- log.(σ_hat)
				D_f = inv(jacob./σ_hat)
				vdelta[pid_t] = vdelta[pid_t] + D_f * f # newton step
			end
			iter[t] += 1

			# update norm
			norm_f = opnorm(f, 1)

			# if inverting demand for first year (max_T = 1 and t = 1), then
			# store the norm between log predicted and observed shares across iterations.
			if max_T == 1 && t == 1
				norm_mat = vcat(norm_mat, hcat(norm_f, iter[t]))
				@printf " t = %f, it = %f, norm = %f\n" t iter[t] norm_f
			end

		end

		# if exceed max iter and still not within tolerance, fill with NaN
		if opnorm(f, 1) > tol
			vdelta[pid_t] = fill(NaN, size(car_pid[t, 2], 1))
		end
	end

	# if inverting demand for first year (max_T = 1 and t = 1), then plot and
	# store the evolution of norms
	if max_T == 1
		plt = plot(norm_mat[2:end, 2], norm_mat[2:end, 1], label = "", xlabel = "iterations",
					ylabel = "norm between log predicted and observed mkt shares",
					yguidefontsize = 8)
	    display(plt)
	    savefig(plt, output_path*method*"_norm_evolution.png")
	end

	return vdelta
end

# ---------------------------------------------------------------------------- #
#   (2) grid search over non-linear parameter
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#   (3) estimate paramter using 2-step GMM
# ---------------------------------------------------------------------------- #
