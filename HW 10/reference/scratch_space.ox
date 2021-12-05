
  /***************************************************************************************/
  /* GMM Estimator */
  /***************************************************************************************/
  decl Q,vLParam,vXi;
  decl vParam=new matrix[columns(mZ)][1];
  decl vParam0=vParam;
  decl step=0;

  /* 2SLS weighting matrix */
  A=invert(mIV'mIV);

  println("/* Plot the iteration process */");
  /* Inversion algorithm */
  iprint=1;
  vParam[0]=0.6;
  /* Contraction mapping */
  inverse(&vDelta0, vParam,0,10^(-12));
   /* Newton */
  vDelta0=vDelta_iia;
  inverse(&vDelta0, vParam,1,10^(-12));
  iprint=0;

  //println("/* Grid Search */");
  decl vGrid=range(0,1,0.1);
  decl mQgrid=new matrix[rows(vParam)][columns(vGrid)];
  for(i=0;i<rows(vParam);i++)
    {
      for(j=0;j<columns(vGrid);j++)
	{
	  vParam[i]=vGrid[j];
	  gmm_obj(vParam,&Q, 0,0);
	  //println("grid: ",Q~vParam');
	  if(Q!=.NaN) {
	    mQgrid[i][j]=-Q;
	  }
	  else {
	    Q=100;
	    vDelta0=vDelta_iia;
	  }
	  vParam[i]=vGrid[mincindex(mQgrid[i][]')];
	}
      DrawXMatrix(i,mQgrid[i][],"Obj",vGrid,"$\\lambda_p$");
    }
  //ShowDrawWindow();
  //SaveDrawWindow(sprint("Car_demand_grid_spec",spec,".pdf"));

  //println("/* Two-step GMM */");
  do{
    vParam0=vParam;
    if(step>0) {
      mG=(vXi.*mIV); mG-=meanc(mG);
      A=invert(mG'mG);
    }
    MaxControl(100,1);

    //MaxSimplex(gmm_obj,&vParam,&Q,constant(1/10,vParam));
    MaxControl(1000,1);
    MaxBFGS(gmm_obj,&vParam,&Q,0,1);
    inverse(&vDelta0, vParam,1,10^(-12));
    vLParam=ivreg(vDelta0,mX,mIV,A);
    vXi=vDelta0-mX*vLParam;
    step+=1;
    //println("norm: ",norm(vParam0-vParam));
  }while(step<2);

  //println("Parameter estimates: ");
  //println("%r",{"price random-coefficient paramter"},vParam);
  //println("%r",aCharactName[find(aCharactName,varlist)],vLParam);