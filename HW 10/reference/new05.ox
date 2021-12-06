

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

