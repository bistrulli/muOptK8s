clear

period=200;
shift=200;
mod=200;
w=@(x)sin(x/(period/(2*pi)))*mod+shift;

x=linspace(0,period,period*10);
y=w(x);
ypred=[];

window=15;

for i=1:length(y)
    if(i>window)
        ypred(i)=mean(diff(y((i-window):i)))*window+y(i);
    else
        ypred(i)=y(i);
    end
end

figure
hold on
plot(x,w(x),"LineWidth",1.5,"LineStyle","--");
plot(x,ypred,"LineWidth",1.5);
grid on;
box on
legend("original","prediction")