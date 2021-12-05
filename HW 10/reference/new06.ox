#include<oxstd.h>
#import <maximize>
#import <solvenle>
#include <oxdraw.h>
#include <oxfloat.h>
#include <oxprob.h>
#import <maxsqp>
#import<blp_func_ps>

main()
{
	decl i,j,k,l;
	decl spec=1;
	decl aCharactName;
	decl mPanelCharact=loadmat(sprint("Car_demand_characteristics_spec",spec,".dta"),&aCharactName);
	
	/* Panel structure */
	n=rows(mPanelCharact);
	vYear=unique(mPanelCharact[][find(aCharactName,"Year")]);
	println(vYear);
	T=columns(vYear);
}