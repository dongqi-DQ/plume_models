function out = plume_model(model_type,T_sfc,qt_sfc,p_sfc,z_sfc,entrain,RH,varargin)

%% Given surface temperature and humidity, and, tropospheric relative humidity
%% This simple plume model predicts the vertical profile of the environmental temperature
%% and consequently the buoyancy of undiluted (CAPE) and weakly-diluted (extreme convection) plumes

%% For zero-buoyancy plume model: model_type == 'zero-buoyancy'
%  environmental temperature is assumed to be neutral to that of bulk-entraining plume

%% For spectal plume model: model_type == 'spectral'
%  environmental temperature is predicted from detraiment temperature of a
%  spectrum of plumes, by dropping the zero-buoayncy and bulk assumption

%% get values of thermodynamic parameters such as air heat capacity Cp
global c
c = model_constants;
    
%% Default Input arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z_top    = 18000;     % Height of the tropopause (m); z_top increases with surface temperature 
                      % considering that MSE_top = MSE_sfc
                      % z_top = (Lv*q_sfc+Cp*(T_sfc-T_top))/g
                    
powerk   = 1.0;       % Power parameter for relationship between entrainment rate and height

deltaz   = 20;        % Vertical grid spacing (m)

ent_fac  = 0.18;      % Constant for computing parameter u in spectral-plume model

eta      = 0.75;      % Constant for computing parameter u in spectral-plume model

% Optional arguments
if nargin >= 8;  z_top      = varargin{1}; end
if nargin >= 9;  powerk     = varargin{2}; end
if nargin >= 10; deltaz     = varargin{3}; end
if nargin >= 11; ent_fac    = varargin{4}; end
if nargin >= 12; eta        = varargin{5}; end

% if use MSE_top = MSE_sfc to compue z_top
% z_top = z_sfc+(c.Lv0*qt_sfc+c.cp*(T_sfc-0.84*252.))/c.g;

% Argument checks:
%              input    name       type      min    max
check_argument(T_sfc  ,'T_sfc'  ,'numeric',0     ,500   );  % K
check_argument(qt_sfc ,'qt_sfc' ,'numeric',0     ,1     );  % kg/kg
check_argument(p_sfc  ,'p_sfc'  ,'numeric',0     ,inf   );  % Pa
check_argument(entrain ,'entrain' ,'numeric',0     ,inf   );
check_argument(RH      ,'RH'      ,'numeric',0     ,1     );
check_argument(z_sfc  ,'z_sfc'  ,'numeric',0     ,z_top );  % m
check_argument(z_top   ,'z_top'   ,'numeric',z_sfc,inf   );
check_argument(powerk  ,'powerk'  ,'numeric',0     ,4     );
check_argument(deltaz  ,'deltaz'  ,'numeric',0     ,200   ); % m 
check_argument(ent_fac ,'ent_fac' ,'numeric',0.    ,1     );
check_argument(eta     ,'eta'     ,'numeric',0.5   ,2     );


%% Height vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z       = z_sfc:deltaz:z_top;

% make sure z_top is a level
if abs(z(end)- z_top ) > 0.1;    z(end+1) = z_top; end

%% Entrainment profile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set entrainment rate as function of z:
% For zero-buoayncy plume model, it means the entraiment rate of the bulk
% plume at this level z, decreasing with height as this function
% For spectral plume model, it defines the entraiment rate of the plume
% that detrains at this level z
ent = 0.001.*entrain*min(1.,max(0.,((z_top-z)/z_top))).^powerk;
% set entrainment rate of the weakly-entrained plume
ent_w = 0.001*entrain*ones(length(z),1)*ent_fac;

%% Initialize variables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p       = zeros(length(z),1);
logp    = zeros(length(z),1);

qv      = zeros(length(z),1);
qsat    = zeros(length(z),1);

T       = zeros(length(z),1);
T_rho   = zeros(length(z),1);

% environmental saturation MSE
h       = zeros(length(z),1);

% environmental mean state 
T_env   = zeros(length(z),1);
q_env   = zeros(length(z),1);
h_env   = zeros(length(z),1);

% undiluted plume (moist adiabat with zero entraiment), according to
% definition of CAPE (convective available potential energy)
T_u     = zeros(length(z),1);
T_rho_u = zeros(length(z),1);
qv_u    = zeros(length(z),1);
h_u     = zeros(length(z),1);
B_u     = zeros(length(z),1);
CAPE_u  = 0.;
T_u(1)  = T_sfc;
qv_u(1) = qt_sfc;

% weakly-entrained plume, as a representation of the extreme convection
T_w     = zeros(length(z),1);
T_rho_w = zeros(length(z),1);
qv_w    = zeros(length(z),1);
h_w     = zeros(length(z),1);
B_w     = zeros(length(z),1);
CAPE_w  = 0.;
T_w(1)  = T_sfc;
qv_w(1) = qt_sfc;

%% Initial conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Inputs
p(1)     = p_sfc;
T(1)     = T_sfc;
qv(1)    = qt_sfc;


% Derived properties for first level
logp(1)  = log(p(1));
h(1)     = calc_MSE(T(1),qv(1),p(1),z(1));
h_u(1)   = h(1);
h_w(1)   = h(1);

% Flag for LCL level
LCL = 0;

%% Integrate model upward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(z)
    
    % Calculate undiluted plume density temperature, buoyancy and CAPE
    T_rho(i) = T(i).*(1+qv(i)./c.eps-qv(i));
    T_rho_u(i) = T_u(i).*(1+qv_u(i)./c.eps-qv_u(i));
    B_u(i) = c.g*(T_rho_u(i)-T_rho(i))/T_rho(i);
    CAPE_u = CAPE_u+B_u(i)*deltaz;
    
    % Calculate weakly-diluted plume density temperature, buoyancy and CAPE
    T_rho_w(i) = T_w(i).*(1+qv_w(i)./c.eps-qv_w(i));
    B_w(i) = c.g*(T_rho_w(i)-T_rho(i))/T_rho(i);
    CAPE_w = CAPE_w+max(B_w(i),0)*deltaz;

    % Calculate plume saturation specific humidity
    qsat(i) = qq_sat(T(i),p(i));
    
    % Calculate environment properties
    T_env(i) = fzero(@(x) calc_Tv(x,RH,p(i))-T_rho(i) ,T(i).*(1+0.61.*RH.*qv(i)));
    
    % Calculate environment humidity based on assumed relative humidity
    e_env = RH.*e_sat(T_env(i));
    q_env(i) = c.eps.*(e_env./(p(i)-e_env.*(1-c.eps)));
    
    % Calculate environment moist static energy
    h_env(i) = calc_MSE(T_env(i),q_env(i),p(i),z(i));
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if i<length(z)
        % No entrainment if unsaturated
        % Include LCL flag to prevent entrainment turning off above LCL
        if (qv(i)-qsat(i)) < 0 && LCL == 0
            ent(i)=0;
            ent_w(i,:,:) = 0;
            z_lcl = i;
        else
            LCL = 1;
        end

        % Step upward - simple Euler method
        if LCL == 1
            
           if z(i)<z_top
               dent = (ent(i)-ent(i-1))/ent(i);
           else
               dent = 0.;
           end
           if strcmp(model_type,'spectral')
              if ent(i)~=0.
                 h(i+1) = h(i) - ent(i)*( h(i) - h_env(i) ) *( z(i+1)-z(i) ) - (h(1) - h(i))*dent/(1+eta*ent(i)*(z(i)-z(z_lcl)));
              else
                 h(i+1) = h(i);
               end

           elseif strcmp(model_type,'zero-buoyancy')
              h(i+1) = h(i) - ent(i).*( h(i) - h_env(i) ) .*( z(i+1)-z(i) );
           end
        else
            h(i+1) = h(i); 
        end
        
        h_u(i+1) = h_u(i);
        
        h_w(i+1) = h_w(i) - ent_w(i).*( h_w(i) - h_env(i) ) .*( z(i+1)-z(i) );
        
        
        logp(i+1) = logp(i) - c.g./(c.Rd.*T_rho(i)).*( z(i+1)-z(i) );
        
        % Calculate pressure
        p(i+1)  = exp(logp(i+1));

        % Calculate Temperature via root finding algorithm
        T(i+1) = fzero(@(x) calc_MSE(x,qv(i),p(i+1),z(i+1))-h(i+1) ,T(i));

 
        % Calculate Undiluted Temperature via root finding algorithm
        T_u(i+1) = fzero(@(x) calc_MSE(x,qv_u(i),p(i+1),z(i+1))-h_u(i+1) ,T_u(i));
        
        % Calculate Weakly Diluted Temperature via root finding algorithm
        T_w(i+1) = fzero(@(x) calc_MSE(x,qv_w(i),p(i+1),z(i+1))-h_w(i+1) ,T_w(i));

        % Calculate specific humidity
        qv(i+1) = min(qq_sat(T(i+1),p(i+1)),qv(i));
        
        % Calculate undilated specific humidity
        qv_u(i+1) = min(qq_sat(T_u(i+1),p(i+1)),qv_u(i));
        
        % Calculate undilated specific humidity
        qv_w(i+1) = min(qq_sat(T_w(i+1),p(i+1)),qv_w(i));
        

    end
end

h_w = max(h_w,h);
T_w = max(T_w,T_env);
T_rho_w = max(T_rho_w,T_rho);
qv_w = max(qv_w,qv);

rhs = qt_sfc/qq_sat(T_sfc,p_sfc);
for i = 1:length(z)
    if i<=z_lcl
        rh = RH+(rhs-RH)*(z(z_lcl)+z_sfc-z(i))/z(z_lcl);
        % Use root finding algorithm with initial guess
        T_env(i) = fzero(@(x) calc_Tv(x,rh,p(i))-T_rho(i) ,T(i).*(1+0.61.*RH.*qv(i)));

        % Calculate environment humidity based on assumed relative humidity
        e_env = rh.*e_sat(T_env(i));
        q_env(i) = c.eps.*(e_env./(p(i)-e_env.*(1-c.eps)));

        % Calculate environment moist static energy
        h_env(i) = calc_MSE(T_env(i),q_env(i),p(i),z(i));
    end
end

h(z>z_top)=nan;
T(z>z_top)=nan;
T_rho(z>z_top)=nan;


out.T_rho=T_rho;out.T=T;out.qv=qv;out.h=h;out.T_env=T_env;out.q_env=q_env;out.h_env=h_env;
out.T_rho_u=T_rho_u;out.T_u=T_u;out.qv_u=qv_u;out.h_u=h_u;out.cape_u=CAPE_u;out.B_u=B_u;
out.T_rho_w=T_rho_w;out.T_w=T_w;out.qv_w=qv_w;out.h_w=h_w;out.cape_w=CAPE_w;out.B_w=B_w;
out.p=p;
out.z=z;
out.ent=ent;

end

function h = calc_MSE(T,qv,p,z)
% Function to calculate moist static energy

% Get constants
global c

% calculate proportions of vapor, liquid and solid
qv = min(qv, qq_sat(T,p));


% calculate moist static energies of components
hd = c.cp.*(T-c.T0)  + c.g.*z;
hv = c.cpv.*(T-c.T0) + c.g.*z + c.Lv0;
% Calculate moist static energy per unit mass of moist air
h = hd.*(1-qv) + qv.*hv;

end

function h = calc_MSE_rh(T,rh,p,z)
% Function to calculate moist static energy

% Get constants
global c

% calculate proportions of vapor, liquid and solid
qv = rh*qq_sat(T,p);


% calculate moist static energies of components
hd = c.cp.*(T-c.T0)  + c.g.*z;
hv = c.cpv.*(T-c.T0) + c.g.*z + c.Lv0;
% Calculate moist static energy per unit mass of moist air
h = hd.*(1-qv) + qv.*hv;

end

function Tv = calc_Tv(T,RH,p)
% Function to calculate virtual temperature at a given relative humidity

% Get constants
global c


% Calculate mixing ratio
es = e_sat(T);
qv = c.eps.*(RH.*es./(p-RH.*es.*(1-c.eps)));

% calculate virtual temperature
Tv = T.*(1+qv./c.eps-qv);

end

function qs = qq_sat(T,p,varargin)
% Function to calculate the saturation specific humidity

% Get constants
global c

% Calculate saturation mixing ratio
es = e_sat(T);
rs = c.eps.*es./(p-es);

% Calculate saturation specific humidity using total water content if given
if nargin==3
    qt = varargin{1};
    rt = qt./(1-qt);
    qs = rs./(1+max(rt,rs));
else
    qs = rs./(1+rs);
end

end

function [es,varargout] = e_sat(T)
% Function to calculate the saturation vapor pressure
%
% The functions are consistent with the constants given in the
% model_constants subroutine. A faster option is available in which the
% saturation curves are approximated as in Bolton (1980).
%


% Get constants
global c

% Thermodynamically consistent definition of saturation curves
% i.e. integral of Clausius-Clapeyron equation with constant heat
% capacities. See Romps (2008).

esl = c.e0.*(T./c.T0).^((c.cpv-c.cpl)./c.Rv).*...
    exp( ( c.Lv0 - c.T0.*(c.cpv-c.cpl) )./c.Rv .* ( 1./c.T0 - 1./T ) );

esi = c.e0.*(T./c.T0).^((c.cpv-c.cpi)./c.Rv).*...
    exp( ( c.Ls0 - c.T0.*(c.cpv-c.cpi) )./c.Rv .* ( 1./c.T0 - 1./T ) );



% If you want slightly faster (~10%) code, use these approximations 
% (Bolton, 1980). Accurate to within 0.5% for T < 310 K.

%esl = 611.2.*exp( 17.67      .* ( T  - 273.15 ) ./ ( T  - 29.65 ) );
%esi = 611.2.*exp( 21.8745584 .* ( T  - 273.15 ) ./ ( T  - 7.66  ) );


% Calculate ice fraction
[fliq,fice] = calc_fice(T);

es = fliq.*esl + fice.*esi;

% Output the liquid and solid vapor pressure separately if required
if nargout>=2; varargout{1} = esl;     end
if nargout>=3; varargout{2} = esi;     end


end

function [fliq,fice] = calc_fice(T)
% Function to calculate the fraction of condensate that is ice.
% All liquid for T > T0
% All ice for T < T_ice
% Linear function in between
global c

fliq = ( T-c.T_ice )./(c.T0-c.T_ice);
fliq(fliq<0) = 0;
fliq(fliq>1) = 1;
fice = 1-fliq;


end

function const = model_constants
% Thermodynamic and other constants for use in the routines


%% Primary Constants

% Gravity
const.g         = 9.81;         % m/s

% Dry air
const.cp        = 1005.7;       % J/K/kg
const.Rd        = 287.04;       % J/K/kg

% Water vapor
const.cpv       = 1870.0;       % J/K/kg
const.Rv        = 461.5;        % J/K/kg

% Liquid water
const.cpl       = 4190.0;       % J/K/kg

% Solid water
const.cpi       = 2106.0;       % J/K/kg


%% Reference values

% Freezing temperature
const.T0        = 273.15;       % K

% Temperature at which all condensate is ice
const.T_ice     = 233.15;       % K

% reference pressures
const.p00 = 100000;             % Pa
const.e0  = 611.2;              % Pa;  This is the saturation vapor pressure at T0


% Latent heats at reference Temperature
const.Lv0       = 2501000.0;    % J/kg
const.Ls0       = 2834000.0;    % J/kg

%% Derived thermodynamic constants
% Changing these will make the thermodynamics inconsistent
const.cv        = const.cp-const.Rd;
const.cvv       = const.cpv-const.Rv;
const.eps       = const.Rd/const.Rv;


end

function check_argument(var,varname,type,varmin,varmax)
% Function to check arguments are correct


if ~isa(var,type);
    error(['Input: ' varname ' must be of type ' type]); 
end

if isnumeric(var)
    if var < varmin || var > varmax; 
        error(['Input: ' varname ' outside of bounds [' num2str(varmin) ',' num2str(varmax) ']']); 
    end
end

end
 
