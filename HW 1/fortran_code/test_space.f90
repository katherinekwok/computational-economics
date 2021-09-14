
program test

double precision, dimension(2) 		::			z = [1.25d0, 0.2d0] ! Stochastic states

double precision 						              :: 			  z_val     ! variable for looping over good bad states
double precision, dimension(2)						:: 			  z_prob    ! variable for looping over good bad states transition probabilities

double precision, dimension(2, 2)	::			t_matrix 		                  ! Declare Transition matrix
t_matrix(1, :)	=  [0.977d0, 0.023d0]																		! Assign first row of transition matrix
t_matrix(2, :)	=  [0.074d0, 0.926d0]																		! Assign second row of transition matrix



!do i_z = 1, size(z)
!
!	z_val = z(i_z) ! Get state (good or bad) using index
!	z_prob = t_matrix(i_z, :)   ! Get transition probabilities given state
!	print *, z_val
!	print *, z_prob
!
!end do

z_prob = t_matrix(1, :)

print *, z
print *, z_prob
print *, dot_product(z, z_prob)


end program test
