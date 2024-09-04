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
#set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
#set_optimizer_attribute(model, "print_level", 0)

jump=[  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1;
        +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  -1  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        -1  +0  +0  +0  +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  -1  +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  -1  +1  +1  +0  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  -1  +1  +0  +0  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  +0  +0  +0  -1  +1  +1  +0  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  +0  +0  +0  +0  -1  +1  +0  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +1  +0  +0  +0  +0  -1  +1  +0  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  +0  -1  +1  +0;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  -1  +0  +0  +0  -1  +1;
        +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +0  +1  +0  +0  -1  +0;
    ];

delta=10^4
alpha=10^-3
maxNC=100
maxNT=1000

MS=["MSauth","MSvalidateid","MSbookflights",
"MSupdateMiles","MScancelbooking","MSgetrewardmiles",
"MSqueryflights","MSviewprofile","MSupdateprofile"]

#params = matread(@sprintf("%s/git/nodejsMicro/src/params.mat",homedir()))
#MU=params["MU"]

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

MU[5]=1.0/0.024734; #XValidate_e;
MU[6]=1.0/0.039025; #XLogin_e;
MU[9]=1.0/0.033605; #XViewProfile_e;
MU[12]=1.0/0.049099; #XUpdateProfile_e;
MU[15]=1.0/0.026019; #XQuery_e;
MU[20]=1.0/0.013853; #XUpdateMiles_e;
MU[23]=1.0/0.012775; #XGetReward_e;
MU[24]=1.0/0.030366; #XBook_e;
MU[29]=1.0/0.025745; #XCancel_e;
MU[30]=1.0/1; #XBrowse_e;

@variable(model,T[i=1:size(jump,1)]>=0)
@variable(model,X[i=1:size(jump,2)]>=0)
@variable(model,C == 0, Param())
@variable(model,NC[2:10]>=0)
@variable(model,NT[2:10]>=0)
Umax=1.0;

#devo sottrarre gli stati che contano il numero di richieste sincrone, altrimneti non si conservano il numero di job
@constraint(model,sum(X[i] for i in [2,4,5,6,8,9,11,12,14,15,17,19,20,22,23,24,26,29,30])==C)

@constraint(model,jump'*T.==0)
@constraint(model,NC.<=maxNC)
@constraint(model,NT.<=maxNT)
#@constraint(model,NC.==[1,1,1,1,1,1,1,1,1])

#--------rate
#-(-a-b+sqrt((-a+b)^2+10^-2))/2;

#p.delta*min(p.NT(2)-(X(3)+X(6)),X(2));
#Tm2=@NLexpression(model,delta*-(-(NT[2]-(X[3]+X[6]))-X[2]+sqrt((-(NT[2]-(X[3]+X[6]))+X[2])^2+alpha))/2)
@variable(model,Tm2>=0)
#@constraint(model,Tm2<=NT[2]-(X[3]+X[6]))
@constraint(model,Tm2<=X[2])
#p.delta*min(p.NT(3)-(X(5)),X(4));
#Tm3=@NLexpression(model,delta*-(-(NT[3]-X[5])-X[4]+sqrt((-(NT[3]-X[5])+X[4])^2+alpha))/2)
@variable(model,Tm3>=0)
@constraint(model,Tm3<=X[4])
#@constraint(model,Tm3<=NT[3]-X[5])
#min(X(5),p.NC(3));

#Tm4=@NLexpression(model,-(-NC[3]-X[5]+sqrt((-NC[3]+X[5])^2+alpha))/2)
@variable(model,Tm4>=0)
@constraint(model,Tm4<=X[5])
@constraint(model,Tm4<=NC[3])

#min(X(6),p.NC(2));
#Tm5=@NLexpression(model,-(-NC[2]-X[6]+sqrt((-NC[2]+X[6])^2+alpha))/2)
@variable(model,Tm5>=0)
@constraint(model,Tm5<=X[6])
@constraint(model,Tm5<=NC[2])

#p.delta*min(p.NT(9)-(X(9)),X(8));
#Tm6=@NLexpression(model,delta*-(-(NT[9]-X[9])-X[8]+sqrt((-(NT[9]-X[9])+X[8])^2+alpha))/2)
@variable(model,Tm6>=0)
#@constraint(model,Tm6<=NT[9]-X[9])
@constraint(model,Tm6<=X[8])

#min(X(9),p.NC(9));
#Tm7=@NLexpression(model,-(-NC[9]-X[9]+sqrt((-NC[9]+X[9])^2+alpha))/2)
@variable(model,Tm7>=0)
@constraint(model,Tm7<=NC[9])
@constraint(model,Tm7<=X[9])

#p.delta*min(p.NT(10)-(X(12)),X(11));
#Tm8=@NLexpression(model,delta*-(-(NT[10]-X[12])-X[11]+sqrt((-(NT[10]-X[12])+X[11])^2+alpha))/2)
@variable(model,Tm8>=0)
#@constraint(model,Tm8<=NT[10]-X[12])
@constraint(model,Tm8<=X[11])

#min(p.NC(10),X(12));
#Tm9=@NLexpression(model,-(-NC[10]-X[12]+sqrt((-NC[10]+X[12])^2+alpha))/2)
@variable(model,Tm9>=0)
@constraint(model,Tm9<=NC[10])
@constraint(model,Tm9<=X[12])

#p.delta*min(p.NT(8)-(X(15)),X(14));
#Tm10=@NLexpression(model,delta*-(-(NT[8]-X[15])-X[14]+sqrt((-(NT[8]-X[15])+X[14])^2+alpha))/2)
@variable(model,Tm10>=0)
#@constraint(model,Tm10<=NT[8]-X[15])
@constraint(model,Tm10<=X[14])

#min(p.NC(8),X(15));
#Tm11=@NLexpression(model,-(-NC[8]-X[15]+sqrt((-NC[8]+X[15])^2+alpha))/2)
@variable(model,Tm11>=0)
@constraint(model,Tm11<=NC[8])
@constraint(model,Tm11<=X[15])

#p.delta*min(p.NT(4)-(X(18)+X(21)+X(24)),X(17));
#Tm12=@NLexpression(model,delta*-(-(NT[4]-(X[18]+X[21]+X[24]))-X[17]+sqrt((-(NT[4]-(X[18]+X[21]+X[24]))+X[17])^2+alpha))/2)
@variable(model,Tm12>=0)
#@constraint(model,Tm12<=NT[4]-(X[18]+X[21]+X[24]))
@constraint(model,Tm12<=X[17])
#p.delta*min(p.NT(5)-(X(20)),X(19));
#Tm13=@NLexpression(model,delta*-(-(NT[5]-X[20])-X[19]+sqrt((-(NT[5]-X[20])+X[19])^2+alpha))/2)
@variable(model,Tm13>=0)
#@constraint(model,Tm13<=NT[5]-(X[20]))
@constraint(model,Tm13<=X[19])
#p.delta*min(p.NT(7)-(X(23)),X(22));
#Tm15=@NLexpression(model,delta*-(-(NT[7]-X[23])-X[22]+sqrt((-(NT[7]-X[23])+X[22])^2+alpha))/2)
@variable(model,Tm15>=0)
#@constraint(model,Tm15<=NT[7]-(X[23]))
@constraint(model,Tm15<=X[22])

#min(p.NC(4),X(24));
#Tm17=@NLexpression(model,-(-NC[4]-X[24]+sqrt((-NC[4]+X[24])^2+alpha))/2)
@variable(model,Tm17>=0)
@constraint(model,Tm17<=NC[4])
@constraint(model,Tm17<=X[24])

#p.delta*min(p.NT(6)-(X(27)+X(28)+X(29)),X(26));
#Tm18=@NLexpression(model,delta*-(-(NT[6]-(X[27]+X[28]+X[29]))-X[26]+sqrt((-(NT[6]-(X[27]+X[28]+X[29]))+X[26])^2+alpha))/2)
@variable(model,Tm18>=0)
@constraint(model,Tm18<=NT[6]-(X[27]+X[28]+X[29]))
@constraint(model,Tm18<=X[26])

#min(X(29),p.NC(6));
#Tm21=@NLexpression(model,-(-NC[6]-X[29]+sqrt((-NC[6]+X[29])^2+alpha))/2)
@variable(model,Tm21>=0)
@constraint(model,Tm21<=NC[6])
@constraint(model,Tm21<=X[29])

#min(p.NC(5),X(20));
#TmGPS1=@NLexpression(model,-(-NC[5]-X[20]+sqrt((-NC[5]+X[20])^2+alpha))/2)
@variable(model,TmGPS1>=0)
@constraint(model,TmGPS1<=NC[5])
@constraint(model,TmGPS1<=X[20])

#min(p.NC(7),X(23));
#TmGPS2=@NLexpression(model,-(-NC[7]-X[23]+sqrt((-NC[7]+X[23])^2+alpha))/2)
@variable(model,TmGPS2>=0)
@constraint(model,TmGPS2<=NC[7])
@constraint(model,TmGPS2<=X[23])

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


@constraint(model,  T[1]==MU[30]*X[30]) #TClient 
@NLconstraint(model,T[2]==delta*Tm2)
@NLconstraint(model,T[3]==delta*Tm3)
@NLconstraint(model,T[4]==Tm4*MU[5]) #TValidate
@NLconstraint(model,T[5]==Tm5*MU[6]) #TLogin
@NLconstraint(model,T[6]==delta*Tm6) 
@NLconstraint(model,T[7]==0.5*Tm7*MU[9])  #TViewProfile
@NLconstraint(model,T[8]==delta*Tm8)
@NLconstraint(model,T[9]==Tm9*MU[12]) #TUpdateProfile
@NLconstraint(model,T[10]==delta*Tm10)
@NLconstraint(model,T[11]==Tm11*MU[15]) #TQuery
@NLconstraint(model,T[12]==delta*Tm12)
@NLconstraint(model,T[13]==delta*Tm13)
@NLconstraint(model,T[14]==0.5*X[18]/(X[18]+X[27])*TmGPS1*MU[20]) #TUpdateMiles
@NLconstraint(model,T[15]==delta*Tm15)
@NLconstraint(model,T[16]==0.5*X[21]/(X[21]+X[28])*TmGPS2*MU[23]) #TGetReward
@NLconstraint(model,T[17]==Tm17*MU[24]) #TBook
@NLconstraint(model,T[18]==delta*Tm18)
@NLconstraint(model,T[19]==X[27]/(X[18]+X[27])*TmGPS1*MU[20]) #TUpdateMiles
@NLconstraint(model,T[20]==X[28]/(X[21]+X[28])*TmGPS2*MU[23]) #TGetReward
@NLconstraint(model,T[21]==0.5*Tm21*MU[29]) #TCancel
@NLconstraint(model,T[22]==0.5*Tm21*MU[29]) #TCancel

U=[T[5]/(NC[2]*MU[6]),#Uauth
T[4]/(NC[3]*MU[5]),#Uvalidate
T[17]/(NC[4]*MU[24]),#Ubook
(T[14]+T[19])/(NC[5]*MU[20]),#UupdateMiles,
(T[21]+T[22])/(NC[6]*MU[29]),#Ucancel
(T[16]+T[20])/(NC[7]*MU[23]),#UgetRewards,
T[11]/(NC[8]*MU[15]),#Uquery
T[7]/(NC[9]*MU[9]),#Uview
T[9]/(NC[10]*MU[12])#Uupdate
]

@variable(model,E_u[i=1:9]>=0)
#@NLconstraint(model,[i=1:length(U)],U[i]<=0.5)
@NLconstraint(model,[i=1:length(U)],E_u[i]>=U[i]-ut)
@NLconstraint(model,[i=1:length(U)],E_u[i]>=-(U[i]-ut))


@constraint(model,X[1]==X[2]+X[3]+X[6])
@constraint(model,X[3]==X[4]+X[5])
@constraint(model,X[7]==X[8]+X[9])
@constraint(model,X[10]==X[11]+X[12])
@constraint(model,X[13]==X[14]+X[15])
@constraint(model,X[16]==X[17]+X[18]+X[21]+X[24])
@constraint(model,X[18]+X[27]==X[19]+X[20])
@constraint(model,X[21]+X[28]==X[22]+X[23])
@constraint(model,X[25]==X[26]+X[27]+X[28]+X[29])

# Set up channels, publisher and subscriber clients
channels=[@sprintf("%s_usr",name)]
subscriber = Client(host=redisHost, port=6379)
redis_cli=Client(host=redisHost, port=6379)

# Begin the subscription
stop_fn(msg) = msg[end] == "close";  # stop the subscription loop if the message matches

println("Listening for messages on channel",channels)
publish(@sprintf("%s_strt",name),"started"; client=redis_cli)

global Ik=0
global stimes=[]
global outfile=string(UUIDs.uuid4())

subscribe(channels...; stop_fn=stop_fn, client=subscriber) do msg
	global logger
    with_logger(logger) do
		#w=parse(Float64, msg[end])
		@info "recMsg" msg[end] 
		w=round(parse(Float64,msg[end]))
		#w=parse(Float64,get("users";client=redis_cli))
		set_value(C,w)
		global stimes
		global outfile

        Psi=0.9
        @objective(model,Max,Psi*(T[1])/(0.76*w)-(1-Psi)*sum(E_u))
	    stime=@elapsed JuMP.optimize!(model)
	    push!(stimes,stime)
	    
	    global status=termination_status(model)
	    if(status!=MOI.LOCALLY_SOLVED && status!=MOI.ALMOST_LOCALLY_SOLVED)
	        error(status)
	    end

		# Tmk=getTr(mongoClient,10,"MSauth")
		# if(typeof(Tmk)!=Nothing)
		# 	global Ik=Ik+((0.71*w)-Tmk)
		# else
		# 	println("error")
		# 	global Ik=0
		# end

		@info "New Replica" MS value.(NC)
		@info "Utiliation" MS value.(U)
		publish(@sprintf("%s_srv",name),@sprintf("%s\$%s",join(MS,";"),join(value.(NC),";")); client=redis_cli)
		
		# matwrite(@sprintf("./data/%s.mat",outfile), Dict(
	    #     "stimes" => stimes
	    # );)
	end
end
