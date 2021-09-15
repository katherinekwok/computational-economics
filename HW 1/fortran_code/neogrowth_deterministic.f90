! ************************************************************************
! Filename : neogrowth_deterministic.f90
!
! Author : Philip Coyle
! Edited by : Katherine Kwok
!
! Date Created : September 7th, 2021
! Date Edited : September 14th, 2021
!
! Description : This program will use dynamic programming techniques to solve
! a simple neoclassical growth model with a two state markov productivity shock.
!
! Commands to run code:
! gfortran -O2 -o neogrowth_deterministic neogrowth_deterministic.f90
! ./neogrowth_deterministic
! ************************************************************************

! ************************************************************************
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! module : params_grid
!
! Description : This module will form the foundation for our program. In it
! we will allocate space for all paramaters used in this program and set up the
! grids to create a discretized state space
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------

module params_grid

implicit none

! -----------------------------------------------------------------------
! *******************DECLARATION OF PARAMETERS AND VARIABLES*************
! -----------------------------------------------------------------------
! Model Parameters
double precision, parameter :: 				cBET 				= 0.99d0 	! Discount Factor
double precision, parameter :: 				cTHETA 			= 0.36d0	! Capital Share
double precision, parameter :: 				cDEL 				= 0.025d0	! Depreciation Rate


! Tolerance level for convergence and max itations
double precision, parameter :: 				tol			 		= 1d-4 	! Convergence Tolerance
integer, parameter 					:: 				max_it 			= 10000 ! Maximum Number of Iterations
integer 										::				it		 			= 1 		! Itation Counter
integer 										:: 				converged		= 0			! Dummy for VFI Convergence

! Stochastic state variables
double precision, dimension(2) 		::			z														  ! Stochastic states
double precision, dimension(2, 2)	::			t_matrix 		                  ! Declare Transition matrix


! -----------------------------------------------------------------------
! ****************************GRID SET**********************************
! -----------------------------------------------------------------------
! Set up for discretizing the state space (Capital Grid)
integer						 				  :: 				i_k, i_kpr																				! Iteration Counters for k_today and k_tomorrow grid
integer						 				  :: 				i_z																				        ! Iteration Counters for z states
integer, parameter 				  :: 				n_k 				= 1000																! Size of k grid
double precision 						:: 				grid_k(n_k)																				! Allocate Space for k grid
double precision, parameter :: 				min_k 			= 0.01d0															! Minimum of k grid
double precision, parameter :: 				max_k 			= 45d0																! Maximum of k grid
double precision, parameter :: 				step_k 			= (max_k - min_k)/(dble(n_k) - 1d0) 	! Step of k grid
double precision					  :: 				k_today
double precision					  :: 				k_tomorrow

! Global variables for Dynamic Programming
double precision 						:: 				c_today
double precision 						:: 				c_today_temp
double precision 						:: 				y_today
double precision 						:: 				k_tomorrow_max
double precision 						:: 				v_today
double precision 						:: 				v_today_temp
double precision 						:: 				v_tomorrow


! Allocating space for Value and Policy Functions
double precision, dimension(n_k, 2) :: 				pf_c ! Allocate Space for Consumption Policy Function for good and bad state
double precision, dimension(n_k, 2) ::    		pf_k ! Allocate Space for Capital Policy Function for good state
double precision, dimension(n_k, 2) ::			  pf_v ! Allocate Space for Value Function for good state

integer 										::			  i_stat   ! Used for writing data after program has run.

end module params_grid



! ************************************************************************
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! program : neogrowth
!
! Description : This program will use dynamic programming techniques to solve
! a simple deterministic neoclassical growth model.
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------

program neogrowth

use params_grid


implicit none

! Begin Computational Timer
integer 										::				beginning, end, rate

call system_clock(beginning, rate)

! Initialize Grids and Policy Functions
call housekeeping()

! Do Value Function Iteration
call bellman()

call system_clock(end)
write(*,*) ""
write(*,*) "******************************************************"
write(*,*) "Total elapsed time = ", real(end - beginning) / real(rate)," seconds"
write(*,*) "******************************************************"

! Write results
call coda()

write(*,*) ""
write(*,*) "**************************************"
write(*,*) "************END OF PROGRAM************"
write(*,*) "**************************************"
write(*,*) ""

end program neogrowth

! ************************************************************************
! ************************************************************************
! ************************************************************************
! **************************** SUBROUTINES *******************************
! ************************************************************************
! ************************************************************************
! ************************************************************************


! ************************************************************************


! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! subroutine : housekeeping
!
! description : Initializes Grids and Policy Functions
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------

subroutine housekeeping()

use params_grid

implicit none

! Discretizing the state space (capital)
do i_k = 1,n_k
	grid_k(i_k) = min_k + (dble(i_k) - 1d0)*step_k
end do

! Setting up Policy Function guesses
do i_k = 1,n_k
	pf_c(i_k, :) 			    = 0d0  ! Initiate for each row (both good and bad states)
	pf_k(i_k, :) 		    	= 0d0
	pf_v(i_k, :)   	      = 0d0


! Assign values for states and transition matrix
z = [1.25d0, 0.2d0]                            ! states
t_matrix(1, :)	=  [0.977d0, 0.023d0]					 ! Assign first row of transition matrix
t_matrix(2, :)	=  [0.074d0, 0.926d0]					 ! Assign second row of transition matrix

end do

return

end subroutine housekeeping




! ************************************************************************


! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! subroutine : bellman
!
! description : Solves the dynamic programming problem for policy and value
! functions.
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------

subroutine bellman()

use params_grid


implicit none

! allocating space for policy function updates
double precision, dimension(n_k, 2) 						:: 				pf_c_up
double precision, dimension(n_k, 2) 						:: 				pf_k_up
double precision, dimension(n_k, 2) 						::				pf_v_up

double precision, dimension(2)					:: 				diff_c             ! variable for differences for good state and bad state
double precision, dimension(2)					:: 				diff_k
double precision, dimension(2)					:: 				diff_v
double precision, dimension(2)					:: 				max_diff



double precision 						    :: 			  z_val     ! variable for looping over good bad states
double precision, dimension(2)	:: 			  z_prob    ! variable for looping over good bad states transition probabilities


converged = 0
it = 1

! Begin Dynamic Programming Algo.
! Continue to iterate while VFI has not converged (converged == 0) and iteration
! counter is less than maximum number of iterations (it < max_it)
do while (converged == 0 .and. it < max_it)

	! Loop over stochastic states
	do i_z = 1, size(z)

		z_val = z(i_z)              ! Get state (good or bad) using index
		z_prob = t_matrix(i_z, :)   ! Get transition probabilities given state

		! Loop over all possible capital states
		do i_k = 1,n_k

			! ***********************************
			! Define the today's state variables
			! ***********************************
			k_today = grid_k(i_k)

			! ******************************************************
			! Solve for the optimal consumption / capital investment
			! ******************************************************
			v_today = -1d10                   ! Set a very low bound for value
			y_today = z_val*k_today**(cTHETA) ! income today with stochastic state value

			! Loop over all possible capital choices
			do i_kpr = 1,n_k
				k_tomorrow = grid_k(i_kpr)

				! Some values are "temp" values because we are searching exhaustively for
				! the capital/consumption choice that maximizes value
				c_today_temp = y_today + (1-cDEL)*k_today - k_tomorrow

				c_today_temp = max(0d0,c_today_temp)

				! calculate value function
				v_today_temp = log(c_today_temp) + cBET* dot_product(pf_v(i_kpr, :), z_prob)

				! if "temp" value is best so far, record value and capital choice
				if (v_today_temp > v_today) then
					v_today = v_today_temp
					k_tomorrow_max = k_tomorrow
				end if

			end do

			k_tomorrow = k_tomorrow_max
			c_today = y_today + (1-cDEL)*k_today - k_tomorrow

	   	! Update policy functions
		  pf_c_up(i_k, i_z) = c_today
		  pf_k_up(i_k, i_z) = k_tomorrow
		  pf_v_up(i_k, i_z) = v_today

		end do ! end of loop over capital states

	end do ! end of stochastic state loop

	! Find the difference between the policy functions and updates
	do i_z = 1, size(z)
		diff_c(i_z)  = maxval(abs(pf_c(:, i_z) - pf_c_up(:, i_z)))
		diff_k(i_z)  = maxval(abs(pf_k(:, i_z) - pf_k_up(:, i_z)))
		diff_v(i_z)  = maxval(abs(pf_v(:, i_z) - pf_v_up(:, i_z)))

		max_diff(i_z) = diff_c(i_z) + diff_k(i_z) + diff_v(i_z)
	end do

	if (mod(it,50) == 0) then
		write(*,*) ""
		write(*,*) "********************************************"
		write(*,*) "At iteration = ", it
		write(*,*) "Max Difference (good) = ", max_diff(1)
		write(*,*) "Max Difference (bad) = ", max_diff(2)
		write(*,*) "********************************************"
	 end if

	if (max_diff(1) < tol .and. max_diff(2)<tol) then
		converged = 1
		write(*,*) ""
		write(*,*) "********************************************"
		write(*,*) "At iteration = ", it
		write(*,*) "Max Difference (good) = ", max_diff(1)
		write(*,*) "Max Difference (bad) = ", max_diff(2)
		write(*,*) "********************************************"
	end if

	it = it+1

	pf_c 		= pf_c_up
	pf_k 		= pf_k_up
	pf_v		= pf_v_up
end do

return

end subroutine bellman

! ************************************************************************


! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! subroutine : coda
!
! description : Writes results to .dat file
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------

subroutine coda()

use params_grid


implicit none

write(*,*) ""
write (*,*) "Writing PFs to DAT file for Good State"
open(unit = 2, file = 'pfs_neogrowth_good.dat', status = 'replace', action = 'write', iostat = i_stat)
200 format(f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x)

do i_k = 1,n_k
     write(2,200) grid_k(i_k), pf_c(i_k, 1), pf_k(i_k, 1), pf_v(i_k, 1)    ! good state
end do

write(*,*) ""
write (*,*) "Writing PFs to DAT file for Bad State"
open(unit = 2, file = 'pfs_neogrowth_bad.dat', status = 'replace', action = 'write', iostat = i_stat)
300 format(f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x)

do i_k = 1,n_k
	write(2,300) grid_k(i_k), pf_c(i_k, 2), pf_k(i_k, 2), pf_v(i_k, 2)		! bad state
end do

return

end subroutine coda
