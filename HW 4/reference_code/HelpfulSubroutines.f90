! ************************************************************************
! Filename : HelpfulSubroutines.f90
!
! Author : Philip Coyle
!
! Date Created : October 3rd, 2021
!
! Description : This document contains helpful subroutines that will be useful in
! solving problem set 4.
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
! subroutine : housekeeping
!
! description : Solves the dynamic programming problem for policy and value
! functions.
! ------------------------------------------------------------------------
! ------------------------------------------------------------------------

subroutine housekeeping()

! Read in ef.dat
open(unit = 142, file = 'ef.dat', action = 'read', status = 'replace', iostat = i_stat)
51324 format(f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15)
do i_t = 1,t_retire - 1
	read(142,51324) eta(i_t)
end do
close(142)

! Read in agg_conesa_krueger_ss.dat
open(unit = 42, file = 'agg_conesa_krueger_ss.dat', action = 'read', status = 'replace', iostat = i_stat)
324 format(f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15)
do i_t = 1,2
	if (i_t == 2) then
		read(42,324) end_Kt, end_Lt, path_int_rate(n_tt), path_wage(n_tt), path_benefits(n_tt)
	elseif (i_t == 1) then
		read(42,324) start_Kt, start_Lt, path_int_rate(1), path_wage(1), path_benefits(1)
	end if
end do
close(42)
path_int_rate = path_int_rate/100d0


step_Kt 			= (end_Kt - start_Kt)/(dble(n_tt) - 1d0)
step_Lt 			= (end_Lt - start_Lt)/(dble(n_tt) - 1d0)
do i_tt = 1,n_tt
	path_Kt(i_tt) = start_Kt + (dble(i_tt) - 1d0)*step_Kt
	path_Lt(i_tt) = start_Lt + (dble(i_tt) - 1d0)*step_Lt
	! write(*,*) path_Kt(i_tt), path_Lt(i_tt)
end do
path_Kt_up(1) = path_Kt(1)
path_Lt_up(1) = path_Lt(1)


! Read in pfs.dat
open(unit = 12, file = 'pfs_conesa_krueger_ss.dat', action = 'read', status = 'replace', iostat = i_stat)
546 format(f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15)
do t = 1,n_t
	do i_z = 1, n_z
		do i_a = 1,n_a
			read(12,546) pf_l(i_a, i_z, t, n_tt), pf_a(i_a, i_z, t, n_tt), pf_v(i_a, i_z, t, n_tt)
		end do
	end do
end do
close(12)

! Read in dist.dat
open(unit = 43, file = 'dist_conesa_krueger_ss.dat', action = 'read', status = 'replace', iostat = i_stat)
243 format(f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15,2x,f25.15)
do t = 1,n_t
	do i_z = 1, n_z
		do i_a = 1,n_a
			i_state = i_a + n_a*(i_z - 1)
			read(43,243) dist_mu(i_state, t, 1)
		end do
	end do
end do
close(43)

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

				! This is how we calculate expectations. Notice that value function are age (t) and transition time (tt) dependent.
				! Continuation value depends on next periods value function which we solved for when we solved the (t+1) problem.
				v_tomorrow = Pr(1)*pf_v(i_ap,1,t+1, i_tt+1) + Pr(2)*pf_v(i_ap,2,t+1, i_tt+1)

				! Store Maximal Value
			end do

			! update policy and value functions

		end do
	end do
end do

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
double precision 						:: 				dist_mu_up(n_a*n_z)

integer 										:: 				row, col
double precision 						:: 				max_diff

! Solve for invariant transition matrix
dist_mu(:,:,i_tt) = 0d0
! We are given intiial conditions for agents asset distribution at age 0.
dist_mu(1,t,i_tt) = !!!!!!!!!! Same as last problem Set
dist_mu(n_a + 1,t,i_tt) = s!!!!!!!!! Same as last problem Set
do t = 1,n_t-1
	! At each time (age) we calculate a new transition matrix. So initalize.
	dist_trans_mat(:,:) = 0d0

	! We figre out the transition matrix in the usual way (Same as PS3).
	! One slight difference, which I will highlight
	do i_z = 1,n_z
		do i_a = 1,n_a

			! Row's correspond what state agent is at today
			a_tomorrow = pf_a(i_a, i_z, t, i_tt)

			do i_zp = 1,n_z
				do i_ap = 1,n_a

					! Cols's correspond what state agent will be at tomorrow (with some probability)
					dist_trans_mat(row, col) = markov(i_z, i_zp)

				end do
			end do

		end do
	end do

	! Calculate the stationary distribution for the next age group by applying the law of motion.
	dist_mu(:,t+1,i_tt+1) = matmul(transpose(dist_trans_mat),dist_mu(:,t,i_tt))*grate
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

Kt_up = 0d0
Lt_up = 0d0

do t = 1,n_t
	do i_z = 1,n_z
		z_today = grid_z(i_z)
		do i_a = 1,n_a
			i_state = i_a + n_a*(i_z - 1)

			l_today = pf_l(i_a, i_z, t, i_tt)
			productivity =  z_today*eta(t)
			mass = dist_mu(i_state, t, i_tt)


			Kt_up = Kt_up + grid_a(i_a)*mass
			Lt_up = Lt_up + l_today*productivity*mass
		end do
	end do
end do

return

end subroutine aggregate_kl

! ************************************************************************
