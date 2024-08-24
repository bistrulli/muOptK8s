using Jedis,Printf,Ipopt,JuMP,MAT,ParameterJuMP,Mongoc,UUIDs,ArgParse,Logging,LogRoller
#include("getTR.jl")

# Define CLI Arg
s = ArgParseSettings()

@add_arg_table! s begin
    "--name"
        help = "The name of the organization"
        arg_type = String
        required = false
        default = "test"
    "--log_path"   
        help = "Logfile path"
        arg_type = String
        required = false
        default = "logs/test/test_opt.log"
    "--ut"
        help = "Utiliation target"
        arg_type = Float64
        required = false
        default = 0.2
end
parsed_args = parse_args(ARGS, s)
name = parsed_args["name"]
log_path = parsed_args["log_path"]
ut = parsed_args["ut"]

logger = RollingLogger(log_path, 512000, 5, Logging.Info);


wdir=pwd()
redisHost="127.0.0.1"
mongoClient = Mongoc.Client(redisHost, 27017)

#model = Model(()->MadNLP.Optimizer(print_level=MadNLP.INFO))
model = Model(Ipopt.Optimizer)
#set_optimizer_attribute(model, "linear_solver", "pardiso")
set_optimizer_attribute(model, "max_iter", 100000)
#set_optimizer_attribute(model, "tol", 10^-10)
set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
#set_optimizer_attribute(model, "print_level", 0)

jump=[+1  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0;
    +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  -1  +0  +0  +0  +0  +0;
    +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +1  +0  +0;
    +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  -1  +0;
    +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +1  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  -1  +1  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0;
    +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
    ];

delta=10^5
maxNC=200
maxNT=200
alpha=10^-40

#MS=["auth","validateid","bookflights","updateMiles","cancelbooking",
#	"getrewardmiles","queryflights","viewprofile","updateprofile"]

MS=["acmeair-auth","acmeair-customer-validateid","acmeair-booking-bookflights","acmeair-customer-updatemiles","acmeair-booking-cancelbooking",
	"acmeair-flight-getrewardmiles","acmeair-flight-queryflights","acmeair-customer-byidget","acmeair-customer-byidpost"]

#params = matread(@sprintf("%s/git/nodejsMicro/src/params.mat",homedir()))
#MU=params["MU"]
#MU=MU*1.0

MU=ones(1,size(jump,2))*-1

# MU[5]=9.2569; #XValidate_e;
# MU[6]=5.5851; #XLogin_e;
# MU[9]=5.6264; #XViewProfile_e;
# MU[12]=3.7859; #XUpdateProfile_e;
# MU[15]=9.0137; #XQuery_e;
# MU[20]=13.1285; #XUpdateMiles_e;
# MU[23]=15.2845; #XGetReward_e;
# MU[24]=5.6010; #XBook_e;
# MU[29]=7.1427; #XCancel_e;
# MU[30]=3.6319; #XBrowse_e;

# MU[5]=1.0/0.024734; #XValidate_e;
# MU[6]=1.0/0.039025; #XLogin_e;
# MU[9]=1.0/0.033605; #XViewProfile_e;
# MU[12]=1.0/0.049099; #XUpdateProfile_e;
# MU[15]=1.0/0.026019; #XQuery_e;
# MU[20]=1.0/0.013853; #XUpdateMiles_e;
# MU[23]=1.0/0.012775; #XGetReward_e;
# MU[24]=1.0/0.030366; #XBook_e;
# MU[29]=1.0/0.025745; #XCancel_e;
# MU[30]=1.0; #XBrowse_e;
MU[1,[9 12 15 22 25 32 36 41 44 48]]=1.0./[1.0 0.039025 0.024734 0.030366 0.013853 0.025745 0.012775 0.026019 0.033605 0.049099]; 

@variable(model,T[i=1:size(jump,1)]>=0)
@variable(model,X[i=1:size(jump,2)]>=0)
@variable(model,C == 0, Param())
@variable(model,NC[2:10]>=0)
#@variable(model,NT[2:10]>=0)

#devo sottrarre gli stati che contano il numero di richieste sincrone, altrimneti non si conservano il numero di job
@constraint(model,sum(X[i] for i in [9 12 15 22 25 32 36 41 44 48  10 14 17 24 29 35 40 43 47])==C)

@constraint(model,jump'*T.<=10^-6)
@constraint(model,jump'*T.>=-10^-6)
#@constraint(model,NC.<=maxNC)
#@constraint(model,NT.<=maxNT)
#@constraint(model,NC[2:10].==1)

#--------rate
#-(-a-b+sqrt((-a+b)^2+10^-2))/2;

#min(X(12),p.NC(2))
Tm2=@NLexpression(model,-(-NC[2]-X[12]+sqrt((-NC[2]+X[12])^2+alpha))/2)
#@variable(model,Tm2>=0)
#@constraint(model,Tm2<=NT[2]-(X[3]+X[6]))
#@constraint(model,Tm2<=X[2])

#p.P_profile2*X(2)/(X(2)+X(3))*p.MU(44)*min(X(44),p.NC(9));
Tm3=@NLexpression(model,delta*-(-NC[9]-X[44]+sqrt((-NC[9]+X[44])^2+alpha))/2)

#min(X(5),p.NC(3));
Tm4=@NLexpression(model,-(-NC[3]-X[5]+sqrt((-NC[3]+X[5])^2+alpha))/2)

#min(X(48),NC(10));
Tm5=@NLexpression(model,-(-NC[10]-X[48]+sqrt((-NC[10]+X[48])^2+alpha))/2)

#min(X(41),p.NC(8));
Tm6=@NLexpression(model,-(-NC[8]-X[41]+sqrt((-NC[8]+X[41])^2+alpha))/2)

#min(X(22),p.NC(4));
Tm7=@NLexpression(model,-(-NC[4]-X[22]+sqrt((-NC[4]+X[22])^2+alpha))/2)

#min(X(32),p.NC(6));
Tm8=@NLexpression(model,-(-NC[6]-X[32]+sqrt((-NC[6]+X[32])^2+alpha))/2)

#min(X(15),p.NC(3));
Tm10=@NLexpression(model,-(-NC[3]-X[15]+sqrt((-NC[3]+X[15])^2+alpha))/2)

#min(X(25),p.NC(5));
Tm13=@NLexpression(model,-(-NC[5]-X[25]+sqrt((-NC[5]+X[25])^2+alpha))/2)

#min(X(36),p.NC(7));
Tm14=@NLexpression(model,-(-NC[7]-X[36]+sqrt((-NC[7]+X[36])^2+alpha))/2)

@constraint(model,  T[1]==MU[9]*X[9])
@NLconstraint(model,T[2]==MU[12]*Tm2)
@NLconstraint(model,T[3]==X[2]/(X[2]+X[3])*MU[44]*Tm3)
@NLconstraint(model,T[4]==X[3]/(X[2]+X[3])*MU[44]*Tm3)
@NLconstraint(model,T[5]==Tm5*MU[48])
@NLconstraint(model,T[6]==MU[41]*Tm6)
@NLconstraint(model,T[7]==Tm7*MU[22])
@NLconstraint(model,T[8]==X[7]/(X[7]+X[8])*MU[32]*Tm8)
@NLconstraint(model,T[9]==X[8]/(X[7]+X[8])*MU[32]*Tm8)
@NLconstraint(model,T[10]==delta*X[10])
@NLconstraint(model,T[11]==Tm10*MU[15])
@NLconstraint(model,T[12]==0)
@NLconstraint(model,T[13]==delta*X[14])
@NLconstraint(model,T[14]==0)
@NLconstraint(model,T[15]==delta*X[17])
@NLconstraint(model,T[16]==MU[25]*X[18]/(X[18]+X[19]+X[30])*Tm12)
@NLconstraint(model,T[17]==MU[25]*X[19]/(X[18]+X[19]+X[30])*Tm12)
@NLconstraint(model,T[18]==X[20]/(X[20]+X[21]+X[31])*MU[36]*Tm13)
@NLconstraint(model,T[19]==X[21]/(X[20]+X[21]+X[31])*MU[36]*Tm13)
@NLconstraint(model,T[20]==0)
@NLconstraint(model,T[21]==delta*X[24])
@NLconstraint(model,T[22]==0)
@NLconstraint(model,T[23]==0)
@NLconstraint(model,T[24]==0)
@NLconstraint(model,T[25]==delta*X[29])
@NLconstraint(model,T[26]==MU[25]*X[30]/(X[18]+X[19]+X[30])*Tm12)
@NLconstraint(model,T[27]==X[31]/(X[20]+X[21]+X[31])*MU[36]*Tm13)
@NLconstraint(model,T[28]==0)
@NLconstraint(model,T[29]==0)
@NLconstraint(model,T[30]==delta*X[35])
@NLconstraint(model,T[31]==0)
@NLconstraint(model,T[32]==0)
@NLconstraint(model,T[33]==0)
@NLconstraint(model,T[34]==delta*X[40])
@NLconstraint(model,T[35]==0)
@NLconstraint(model,T[36]==delta*X[43])
@NLconstraint(model,T[37]==0)
@NLconstraint(model,T[38]==0)
@NLconstraint(model,T[39]==delta*X[47])
@NLconstraint(model,T[40]==0)



global Ik=0
global stimes=[]
global outfile=string(UUIDs.uuid4())


with_logger(logger) do

	w=10
	#w=parse(Float64,get("users";client=redis_cli))
	set_value(C,w)
	global stimes
	global outfile

    #@objective(model,Max,(T[1]))
    #@objective(model,Max,1.0*T[1]-0.0*(sum(NC)/(maxNC*9)))
    stime=@elapsed JuMP.optimize!(model)
    push!(stimes,stime)
    
    global status=termination_status(model)
    if(status!=MOI.LOCALLY_SOLVED && status!=MOI.ALMOST_LOCALLY_SOLVED)
        error(status)
    end

    println(value.(T))

	@info "New Replica" MS value.(NC)
	publish(@sprintf("%s_srv",name),@sprintf("%s\$%s",join(MS,";"),join(value.(NC),";")); client=redis_cli)
	
	# matwrite(@sprintf("./data/%s.mat",outfile), Dict(
    #     "stimes" => stimes
    # );)
end


#--------------
# npoint=40
# NCopt=zeros(9,npoint)
# NTopt=zeros(9,npoint)
# stimeOpt=zeros(1,npoint)
# clients=rand(1,npoint)'*500
# #clients=LinRange(1,100, npoint);
# #clients=[1]
#
# for i=1:size(clients,1)
#     global w=round(clients[i])
#     set_value(C,w)
#
#     @objective(model,Max,0.5*(T[1])*15/(w)-0.5*(sum(NC)+sum(NT))/(maxNC*10+maxNT*10))
#     global stimes=@elapsed JuMP.optimize!(model)
#     global status=termination_status(model)
#     if(status!=MOI.LOCALLY_SOLVED && status!=MOI.ALMOST_LOCALLY_SOLVED)
#         error(status)
#     end
#
#     #RTv=[value(X[1]+X[5])/value(T[1]),value(X[3])/value(T[4]),value(X[4])/value(T[5])];
#     #Tv=[value(T[1]),value(T[4]),value(T[5])]
#
#     NCopt[:,i]=value.(NC)
#     NTopt[:,i]=value.(NT)
#     stimeOpt[i]=stimes
# end
#
# matwrite("re.mat", Dict(
# 	"NC_opt" => NCopt,
# 	"NT_opt" => NTopt,
# 	"Clients" =>  collect(clients),
# 	"rtime_opt" => stimeOpt
# );)
