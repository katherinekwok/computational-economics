! ************************************************************************
! Filename : HelpfulSubroutines.f90
!
! Author : Philip Coyle
!
! Date Created : September 27th, 2020
!
! Description : This document contains helpful subroutines that will be useful in
! solving problem set 3.
! ************************************************************************


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


! Loop over ages backward
do t = n_t,-1,1
	! Loop over states
	do i_z = 1,n_z
		! Define today's shock
		Pr = markov(i_z,:)

		do i_a = 1,n_a

			do i_ap = 1,n_a
				a_tomorrow = grid_a(i_ap)

				! Imagine there is the appropriate code here to solve agent's problem
				! Note that there are 3 separate cases to consider
				! (1) End of life (no continuation value)
				! (2) Retirees (get positive benefits and don't work)
				! (3) Workers (work and don't get benefits)

				! This is how we calculate expectations. Notice that value function are time (age) dependent.
				! Continuation value depends on next periods value function which we solved for when we solved the (t+1) problem.
				v_tomorrow = Pr(1)*pf_v(i_ap,1,t+1) + Pr(2)*pf_v(i_ap,2,t+1)

				! Store Maximal Value
			end do

			! update policy and value functions

		end do
	end do
end do


return

end subroutine bellman

! ************************************************************************


! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! subroutine : ss_dist
!
! description : Solves the stationary wealth distribution.
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------


subroutine ss_dist()

use params_grid

implicit none

! allocating space for policy function updates
integer 										:: 				row, col
double precision 						:: 				max_diff

! Solve for invariant transition matrix
dist_mu(:,:) = 0d0

! We are given intiial conditions for agents asset distribution at age 0.
dist_mu(1,t) = ss_zg/tot_t
dist_mu(n_a + 1,t) = ss_zb/tot_t
do t = 1,n_t-1
	! At each time (age) we calculate a new transition matrix. So initalize.
	dist_trans_mat(:,:) = 0d0

	! We figre out the transition matrix in the usual way (Same as PS2).
	do i_z = 1,n_z
		do i_a = 1,n_a

			! Row's correspond what state agent is at today
			row = i_a + n_a*(i_z - 1)
			a_tomorrow = pf_a(i_a, i_z, t)

			do i_zp = 1,n_z
				do i_ap = 1,n_a
					if (a_tomorrow == grid_a(i_ap)) then

						! Cols's correspond what state agent will be at tomorrow (with some probability)
						col = i_ap + n_a*(i_zp - 1)
						dist_trans_mat(row, col) = markov(i_z, i_zp)

					end if
				end do
			end do

		end do
	end do

	! Calculate the stationary distribution for the next age group by applying the law of motion.
	dist_mu(:,t+1) = matmul(transpose(dist_trans_mat),dist_mu(:,t))*grate
end do


return


end subroutine ss_dist


! ************************************************************************


! ------------------------------------------------------------------------
! ------------------------------------------------------------------------
! subroutine : aggregate_kl
!
! description : Solves for aggregate capital and labor supply.
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------


subroutine aggregate_kl()

use params_grid

implicit none
double precision 						:: 				productivity
double precision 						:: 				mass

K_agg_up = 0d0
L_agg_up = 0d0

do t = 1,n_t
	do i_z = 1,n_z
		z_today = grid_z(i_z)
		do i_a = 1,n_a
			i_state = i_a + n_a*(i_z - 1)

			a_today = grid_a(i_a)
			l_today = pf_l(i_a, i_z, t)
			productivity =  z_today*eta(t)
			mass = dist_mu(i_state, t)

			K_agg_up = K_agg_up + a_today*mass
			L_agg_up = L_agg_up + l_today*productivity*mass
		end do
	end do
end do

return

end subroutine aggregate_kl
