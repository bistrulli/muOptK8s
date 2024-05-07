using Printf,Ipopt,JuMP,MAT,ParameterJuMP,Statistics,Jedis,ArgParse,Logging,LogRoller

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
        help = "utilizzo target"
        arg_type = Float64
        required = false
        default = 0.5
end
parsed_args = parse_args(ARGS, s)
name = parsed_args["name"]
log_path = parsed_args["log_path"]
ut=parsed_args["ut"]

logger = RollingLogger(log_path, 512000, 5, Logging.Info);

wdir=pwd()

#model = Model(()->MadNLP.Optimizer(print_level=MadNLP.INFO))
model = Model(Ipopt.Optimizer)
#set_optimizer_attribute(model, "linear_solver", "pardiso")
set_optimizer_attribute(model, "max_iter", 100000)
#set_optimizer_attribute(model, "tol", 10^-10)
set_optimizer_attribute(model, "hessian_approximation", "limited-memory")
set_optimizer_attribute(model, "print_level", 0)
        #Xb Xd Xb1 X1e  x12 x2e x23 x3e
jump=[  -1  +1  +0  +0  +0  +0  +0  +0;
        +0  -1  +1  +1  +0  +0  +0  +0;
        +0  +0  +0  -1  +1  +1  +0  +0;
        +0  +0  +0  +0  +0  -1  +1  +1;
        +1  +0  -1  +0  -1  +0  -1  -1;
    ];

alpha=10^-4
maxNC=100

#params = matread(@sprintf("%s/git/nodejsMicro/src/params.mat",homedir()))
#MU=params["MU"]

MU=ones(1,size(jump,2))*-1

R=zeros(1,3)
dimRep=1

MU[1]=0.9709; #XValidate_e;
MU[2]=100;
MU[3]=18
MU[4]=18
MU[5]=18

@variable(model,T[i=1:size(jump,1)]>=0)
@variable(model,X[i=1:size(jump,2)]>=0)
@variable(model,C == 10, Param())
@variable(model,NC[1:3]>=0)
@variable(model,E_ss[i=1:size(jump,2)]>=0)
@variable(model,E_u[i=1:size(NC,1)]>=0)

#devo sottrarre gli stati che contano il numero di richieste sincrone, altrimneti non si conservano il numero di job
@constraint(model,sum(X[i] for i in [1,2,4,6,8])==C)

@constraint(model,E_ss.>=jump'*T)
@constraint(model,E_ss.>=-(jump'*T))
@constraint(model,E_ss.<=10^-6)
#@constraint(model,jump'*T.==0)
#@constraint(model,NC.<=maxNC)

#--------rate
#min(a,b)=-(-a-b+sqrt((-a+b)^2+10^-2))/2;

Tm1=@expression(model,X[1])
Tm2=@expression(model,X[2])

#min(X(4),NC(1));
Tm3=@NLexpression(model,-(-NC[1]-X[4]+sqrt((-NC[1]+X[4])^2+alpha))/2)
#min(X(6),NC(2));
Tm4=@NLexpression(model,-(-NC[2]-X[6]+sqrt((-NC[2]+X[6])^2+alpha))/2)
#min(X(8),NC(3));
Tm5=@NLexpression(model,-(-NC[3]-X[8]+sqrt((-NC[3]+X[8])^2+alpha))/2)

@constraint(model,T[1]==MU[1]*Tm1)
@constraint(model,T[2]==MU[2]*Tm2)
@NLconstraint(model,T[3]==MU[3]*Tm3)
@NLconstraint(model,T[4]==MU[4]*Tm4)
@NLconstraint(model,T[5]==MU[5]*Tm5)


@constraint(model,X[3]==X[4]+X[5])
@constraint(model,X[5]==X[6]+X[7])
@constraint(model,X[7]==X[8])


@constraint(model,E_u[1]>=(ut*MU[3]*NC[1]-T[3]))
@constraint(model,E_u[1]>=-(ut*MU[3]*NC[1]-T[3]))


@constraint(model,E_u[2]>=(ut*MU[4]*NC[2]-T[4]))
@constraint(model,E_u[2]>=-(ut*MU[4]*NC[2]-T[4]))

@constraint(model,E_u[3]>=(ut*MU[5]*NC[3]-T[5]))
@constraint(model,E_u[3]>=-(ut*MU[5]*NC[3]-T[5]))

#conn = RedisConnection()

#function message_callback(msg)
channels=[@sprintf("%s_usr",name)]
println("started")

# Begin the subscription
stop_fn(msg) = msg[end] == "close";

redis_cli = Client(host="localhost", port=6379)
subscriber = Client(host="localhost", port=6379)

println("Listening for messages on channel",channels)
publish(@sprintf("%s_strt",name),"started"; client=redis_cli)

subscribe(channels...; stop_fn=stop_fn, client=subscriber) do msg
    global logger
    with_logger(logger) do
        # Process the received message here
        @info "recMsg" msg[3] 
        w=round(parse(Float64,msg[3]))
        set_value(C,w)

        #@objective(model,Max,T[1]-sum(E_u))
        @objective(model,Min,sum(E_u))
        global stimes=@elapsed JuMP.optimize!(model)
        global status=termination_status(model)
        if(status!=MOI.LOCALLY_SOLVED && status!=MOI.ALMOST_LOCALLY_SOLVED)
            error(status)
        end

        #scendo il numero di repliche solo se occorre veramente
        slack=0
        util=zeros(1,length(NC))
        for tier=1:length(NC)
            nR=value(NC[tier])/dimRep
            @info "tier" tier "rawReplica" nR
            if(R[end,tier]>ceil(nR))#downscaling
                @info "downscaling"
                if(nR-floor(nR)>=0.10)
                    slack=0.9
                    logmsg=@sprintf("dovrei andare a %d invece mantengo %d",ceil(nR),ceil(nR+slack))
                    @info logmsg
                end
            elseif(R[end,tier]<ceil(nR))#upscaling
                @info "upscaling"
                #if(nR-floor(nR)>=0.8)
                if(false)
                    slack=+0.21
                    logmsg=@sprintf("dovrei andare a %d invece mantengo %d",ceil(nR),ceil(nR+slack))
                    @info logmsg
                end
            end
            global R[end,tier]=max(ceil(nR+slack),1)
            global util[1,tier]=value(T[2+tier]/(MU[2+tier]*NC[tier]))
        end
        @info "New Replica" R[end,:]
        @info "Util" util
        publish(@sprintf("%s_srv",name),join(R[end,:],"-"); client=redis_cli)
    end
end



# # Subscribe to a Redis channel
# channel_name = @sprintf("%s_usr",name)

# sub = open_subscription(conn)
# subscribe_data(sub, channel_name, message_callback)

# # Keep the Julia process running to receive messages
#println("Listening for messages on channel $channel_name...")
# publish(conn, @sprintf("%s_strt",name),"started")
# sleep(1000000)  # Sleep indefinitely to keep the process alive
