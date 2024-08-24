using Jedis,Printf,Ipopt,JuMP,MAT,ParameterJuMP,Mongoc,UUIDs,ArgParse,Logging,LogRoller

minAlfa=10^-30
ρ=30
#fatto in questo modo va bene per il caso singola classe
#come faccio a gestire il caso GPS sul numero di chiamate?
function T(X,NC,MU,J,P)
    i=findall(x->x==-1,J)[1]
    j=findall(x->x==1,J)[1]
    #gps=@NLexpression(model,-(-NC[i]-X[i]+sqrt((-NC[i]+X[i])^2+minAlfa))/2)
    #min(a, b) ≈ 0.5 * (a + b - (a - b) * tanh(ρ * (a - b)))
    gps=@NLexpression(model,0.5 * (NC[i] + X[i] - (NC[i] - X[i]) * tanh(ρ * (NC[i] - X[i]))))

    
    [@NLexpression(model,MU[i]*gps*P[i,j])] 
end

jump=[-1  +1  +0  +0;
      -1  +0  +1  +0;  
	  +0  -1  +0  +1;
      +0  +0  -1  +1;
      +1  +0  +0  -1;
	  ]

P=[0.  .5  .5  0.;
   0.  0.  0.  1.;
   0.  0.  0.  1.;
   1.  0.  0.  0.;]


H=20
dt=0.01

#NC=[1000. 1. 1.]
MU=[1. 1. 1. 1.]
X0=[0.  0.  0.  10.]

ref=ones(H,1)*2

model = Model(Ipopt.Optimizer)
#set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
set_optimizer_attribute(model, "max_iter", 10000)

@variable(model,Xref[i=1:size(ref,1)]>=0)
@variable(model,Eabs[i=1:size(ref,1)]>=0)

@variable(model,NC[i=1:size(jump,2),h=1:H-1]>=0.001)
@constraint(model,[i=2:size(jump,2)],NC[i,:].<=100)
@constraint(model,NC[1,:].==1000)

#@constraint(model,NC[2:size(jump,2)].==ones(1,size(jump,2)-1))
@constraint(model,Eabs.>=Xref-ref)
@constraint(model,Eabs.>=-(Xref-ref))
    

X=Array{Union{Nothing,NonlinearExpression,Float64}}(nothing, size(jump,2), H)
X[:,1]=X0'
#integro equazioni
for h=1:H-1
    for i=1:size(jump,2)
        jRow=jump'[i,:]
        dXih=@NLexpression(model,sum(jRow[j]*T(X[:,h],NC[:,h],MU,jump[j,:],P)[1] for j=1:size(jRow,1)))
        X[i,h+1]=@NLexpression(model,dXih*dt+X[i,h])
    end
end


@NLconstraint(model,optCon[i=1:size(Xref,1)],Xref[i]==X[1,i])
@objective(model,Min,sum(Eabs)+0.001*sum(NC[i,h] for i=2:size(jump,2) for h=1:H-1 ))

stime=@elapsed JuMP.optimize!(model)
status=termination_status(model)
if(status!=MOI.LOCALLY_SOLVED && status!=MOI.ALMOST_LOCALLY_SOLVED)
   error(status)
end

display(value.(X[:,end])')



