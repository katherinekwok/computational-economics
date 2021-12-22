/* MLE with true CCP */
  vPhat=vP;
  MaxControl(1000,1);
  MaxBFGS(lfunc_ccp,&vParam,&L,0,1);
  decl vParam_true_ccp=vParam;
  
  /* Nested Fixed Point */
  decl vParam_est_nfxp=vParam_true;
  vParam_est_nfxp[ilambda]=lambda;
  MaxBFGS(lfunc_nfxp,&vParam_est_nfxp,&L,0,1);
  //MaxSimplex(lfunc_nfxp,&vParam_est_nfxp,&L,constant(.25,vParam_est_nfxp));

  println("////////////////////////////////////////////////////");
  println("Estimation results: ");
  println("%r",{"lambda"},"%c",{"true","2-step","True CCP","NFXP"},vParam_true~vParam_est_ccp~vParam_true_ccp~vParam_est_nfxp);


  /***************************************/
  // plot lambda value
  decl vGrid_lambda=range(-10,0,1/10);
  decl vGrid_llf=<>;  
  for(i=0;i<columns(vGrid_lambda);i++)
    {
      vParam_est_nfxp[ilambda]=vGrid_lambda[i];
      lfunc_nfxp(vParam_est_nfxp,&L,0,0);
      vGrid_llf~=L/Sim;
    }
  DrawXMatrix(0,vGrid_llf,"llf",vGrid_lambda,"lambda",0,5);  
  ShowDrawWindow();
  SaveDrawWindow("grid_search_lambda.pdf");
   /***************************************/
  