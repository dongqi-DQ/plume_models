function [T_rho,T,p,qv,qsat,ql,qi,z,h,T_env,q_env,h_env,ent] = zero_buoyancy_plume(T_base,qt_base,p_base,entrain,RH,varargin)
%ZERO_BUOYANCY_PLUME calculate zero-buoyancy plume thermodynamic profile
%
% This is the zero-buoyancy plume model used in Singh & O'Gorman (2013).
% The model solves for the temperature and humidity profile of an
% entraining plume that is neutrally buoyant with respect to its
% environment given a base temperature and humidity and an
% environmental relative humidity.
%
% %% USAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Simply running the script with no input arguments will solve the model
% with some example parameters and produce a few plots.
%
% The full calling syntax is:
%
% [T_rho,T,p,qv,qsat,ql,qi,z,s,T_env,q_env,s_env,ent] = ...
%            zero_buoyancy_plume(                ...
%                 T_init,qv_init,p_init,entrain,RH,     ...
%                [z_bot,z_top,gamma,ent_type,deltaz,T_ice])
%
% ARGUMENTS:
%           T_base  :  base temperature of plume            (K)
%           qt_base :  base specific humidity of plume      (kg/kg)
%           p_base  :  base pressure pf plume               (Pa)
%           entrain :  entrainment parameter                (unitless)
%                        Dependent on "ent_type"
%                        entrainment = entrain/z            [default]
%                        entrainment = entrain/1000
%           RH      :  environmental relative humidity      (0-1)
%
% OPTIONAL ARGUMENTS:
%           z_base  :  base height of plume                 (m)    [50]
%           z_top   :  height to which plume is integrated  (m)    [15000]
%           gamma   :  fraction of water that precipitates  (0-1)  [1]
%                       0 = no fallout
%                       1 = pseudo-adiabatic
%
%           ent_type:  type of entrainment profile                 ['invz']
%                        'invz' : entrainment = entrain/z
%                        'const': entrainment = entrain/1000
%           deltaz  :  vertical grid-spacing                (m)    [50]
%           T_ice   :  Temperature of total freezing        (K)    [233.15]
%                        Condensate freezes gradually
%                        between 273.15 K and T_ice.
%
%
% OUTPUTS:
%           T_rho   : Density temperature (plume and env)   (K)
%           T       : Temperature of plume                  (K)
%           p       : pressure            (plume and env)   (K)
%           qv      : specific humidity of plume            (kg/kg)
%           qsat    : saturation specific humidity of plume (kg/kg)
%           ql      : liquid water mass fraction of plume   (kg/kg)
%           qi      : solid water mass fraction of plume    (kg/kg)
%           z       : height                                (z)
%           h       : plume moist static energy             (J/kg)
%           T_env   : Temperature of environment            (K)
%           q_env   : specific humidity of environment      (kg/kg)
%           h_env   : environment moist static energy       (J/kg)
%           ent     : entrainment rate                      (1/m)
%
%
%
% %% MODEL DEATILS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% The entraining plume model solves the equations
%
% dh/dz      = -epsilon (h   - h_env)
% dq_t/dz    = -epsilon (q_t - q_env)
% dlog(p)/dz = - g/(R_d T_rho)
%
% where h is the moist static energy, q_t is the total water content, p is 
% the pressure, and T_rho is the density temperature.
%
% The moist static energies are defined:
%
% h_d = c_pd (T -T0)  + gz
% h_v = c_pv (T -T0)  + gz + Lv0 q_v
% h_l = c_l  (T -T0)  + gz
% h_i = c_i  (T -T0)  + gz - Lf0 q_i
% h   = (1-q_t)h_d + h_v s_v + q_l h_l + q_i h_i
%
% The equations are integrated in height using Euler forward differences for
% the entrainment terms. Precipitation fallout is handled separately.
% After each Euler step a fraction \gamma of the increase in liquid and
% solid water in the plume is discarded, and the moist static energy
% adjusted consistently.
%
% The environmental properties are calculated by assuming the density of
% the plume and parcel are equal, and using the given environmental
% relative humidity.
%
% Full details are found in the supplementary information of
% Singh & O'Gorman (2013). The equations used are consistent with the
% parcel model of Romps & Kuang (2010) applied to a vertically oriented,
% steady flow in the limit of zero cloud buoyancy.
%
%
%
% %% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Bolton, D. (1980). The Computation of Equivalent Potential Temperature.
% Mon. Wea. Rev., 108, 1046-1053.
%
% Romps, D.M. (2008). The Dry-Entropy Budget of a Moist Atmosphere.
% J. Atmos. Sci., 65, 3779-3799.
%
% Romps, D.M. & Kuang, Z (2010). Do Undiluted Convective Plumes Exist in
% the Upper Tropical Troposphere?. J. Atmos. Sci., 67, 468-484.
%
% Singh, M.S. & O'Gorman (2013). Influence of entrainment on the thermal
% stratification in simulations of radiative-convective equilibrium.
% Geophs. Res. Lett, 40, doi:10.1002/grl.50796.
%
% %% AUTHOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Written by Martin S. Singh, 10 Jul 2013.
%            mssingh@mit.edu
%
% 
%
% %% UPDATE HISTORY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   
%   16 Jul 2013: MS 
%       - Added argument checks
%       - Fixed bug in specifying total freezing temperature (T_ice)
%       - Cleaned up some typos in comments
%
%   28 Aug 2013: MS 
%       - updated reference to Singh & O'Gorman as article now published
%
%


%% Load constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global c
c = model_constants;

%% Demonstration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin == 0
    T_base  = 300;
    qt_base = 0.0155;
    p_base  = 100000;
    entrain = 0.5;
    RH      = 0.7;
end

%% Default Input arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z_base = 50;        % Height of plume initial condition    (m)
z_top = 15000;      % Height to which plume is integrated  (m)
gamma = 1;          % Fraction of condensation that falls out as precipitation
ent_type = 'invz' ; % Type of entrainment profile
                    % 'invz'  1/z profile;                entrainment_rate = entrain/z
                    % 'const' constant in z profile;      entrainment_rate = entrain/1000

deltaz = 50;        % Vertical grid spacing                (m)


% Optional arguments
if nargin >= 6;  z_base     = varargin{1}; end;
if nargin >= 7;  z_top      = varargin{2}; end;
if nargin >= 8;  gamma      = varargin{3}; end;
if nargin >= 9;  ent_type   = varargin{4}; end;
if nargin >= 10; deltaz     = varargin{5}; end;
if nargin >= 11; c.T_ice    = varargin{6}; end;

% Argument checks:
%              input    name       type      min    max
check_argument(T_base  ,'T_base'  ,'numeric',0     ,500   );
check_argument(qt_base ,'qt_base' ,'numeric',0     ,1     );
check_argument(p_base  ,'p_base'  ,'numeric',0     ,inf   );
check_argument(entrain ,'entrain' ,'numeric',0     ,inf   );
check_argument(RH      ,'RH'      ,'numeric',0     ,1     );
check_argument(z_base  ,'z_base'  ,'numeric',0     ,z_top );
check_argument(z_top   ,'z_top'   ,'numeric',z_base,inf   );
check_argument(gamma   ,'gamma'   ,'numeric',0     ,1     );
check_argument(ent_type,'ent_type','char'   ,0     ,500   );
check_argument(deltaz  ,'deltaz'  ,'numeric',0     ,inf   );
check_argument(c.T_ice ,'T_ice'   ,'numeric',0     ,c.T0  );





%% Height vector %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z       = z_base:deltaz:z_top;

% make sure z_top is a level
if abs(z(end)- z_top ) > 0.1;    z(end+1) = z_top; end

%% Entrainment profile %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set entrainment rate as function of z
switch ent_type
    case 'invz'
        % Entrainment rate  m^-1
        ent     = min(1e-2,entrain./z);
    case 'const'
        % Entrainment rate  m^-1
        ent = 0.001.*entrain.*ones(size(z));
    otherwise
        error('unknown entrainment profile type')
end

%% Initialize verctors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p       = zeros(length(z),1);
logp    = zeros(length(z),1);

qv      = zeros(length(z),1);
qsat    = zeros(length(z),1);
ql      = zeros(length(z),1);
qi      = zeros(length(z),1);
qt      = zeros(length(z),1);

T       = zeros(length(z),1);
T_rho   = zeros(length(z),1);

h       = zeros(length(z),1);

T_env   = zeros(length(z),1);
q_env   = zeros(length(z),1);
h_env   = zeros(length(z),1);


%% Initial conditions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Inputs
p(1)     = p_base;
T(1)     = T_base;
qt(1)    = qt_base;


% Determine if there is any liquid or solid water
[qv(1),ql(1),qi(1)] = calc_saturation(p_base,T_base,qt_base);
if ql(1)>0 || qi(1) > 0; warning('PLUME:SuperSat','plume base super-saturated'); end

% Derived properties
logp(1)  = log(p(1));
h(1)     = calc_MSE(T(1),qt(1),p(1),z(1));

% Flag for LCL level
LCL = 0;


%% Integrate model upward %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(z)
    
    
    % Calculate plume density temperature
    T_rho(i) = T(i).*(1+qv(i)./c.eps - qt(i));
    
    % Calculate plume saturation specific humidity
    qsat(i) = qq_sat(T(i),p(i),qt(i));
    
    
    % Calculate environment properties
    % Assume:
    %           1) equal density to plume
    %           2) Fixed relative humidity
    
    % Tv(env) = T_rho(plume)
    % Use root finding algorithm with initial guess
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
        if (qt(i)-qsat(i)) < 0 && LCL == 0;
            ent(i)=0;
        else
            LCL = 1;
        end
        
        % Step upward - simple Euler method
        h(i+1) =    h(i) - ent(i).*( h(i) - h_env(i) ) .*( z(i+1)-z(i) );
        qt(i+1) =   qt(i) - ent(i).*( qt(i)-q_env(i) )  .*( z(i+1)-z(i) );
        logp(i+1) = logp(i) - c.g./(c.Rd.*T_rho(i))       .*( z(i+1)-z(i) );
        
        % Calculate pressure
        p(i+1)  = exp(logp(i+1));
        
        % Calculate Temperature via root finding algorithm
        T(i+1)  = fzero(@(x) calc_MSE(x,qt(i+1),p(i+1),z(i+1))-h(i+1) ,T(i));
        
        % Calculate humidity
        [qv(i+1),ql(i+1),qi(i+1)] = calc_saturation(p(i+1),T(i+1),qt(i+1));
        
        
        % Rainfall fallout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        if qt(i+1)-qv(i+1)>0 && gamma > 0;
            
            if gamma>=1 % Total fallout:
                
                %   1) Set q to its saturation value
                qt(i+1) = qq_sat(T(i+1),p(i+1));
                [qv(i+1),ql(i+1),qi(i+1)] = calc_saturation(p(i+1),T(i+1),qt(i+1));
                
                %   2) Recalculate moist static energy
                h(i+1)  = calc_MSE(T(i+1),qt(i+1),p(i+1),z(i+1));
                
            else        % Partial fallout:
                
                % Calculate increase in liquid/solid water
                % Assume fallout is proportional to this value
                dqls = -(ql(i+1)+qi(i+1)-ql(i)-qi(i));
                
                % Calculate fallout
                [fl,fs] = fallout(gamma,ql(i+1),qi(i+1),dqls,T(i+1));
                
                % Minimum value of total water is saturation value
                qtmin = qq_sat(T(i+1),p(i+1));
                
                % Adjust total water, prevent reduction below saturation
                qt(i+1) = max(qt(i+1) - (fl+fs).*(1-qt(i+1)),qtmin);
                
                % Adjust moist static energy
                h(i+1)  = calc_MSE(T(i+1),qt(i+1),p(i+1),z(i+1));
                
                % Recalculate water species breakdown
                [qv(i+1),ql(i+1),qi(i+1)] = calc_saturation(p(i+1),T(i+1),qt(i+1));
            end
            
        end
    end
end


if nargin ==0 % Demonstration plot
    figure;
    set(gcf,'defaultaxeslinewidth',1,'defaultaxesfontsize',14)
    
    % Left axis
    axes('position',[0.1 0.12 0.4 0.72]);
     plot(T_rho,z./1000,'r','linewidth',2)
     xlabel('T_\rho (K)')
     ylabel('z (km)')
     set(gca,'xtick',[200 250 300],'xlim',[180 310])
     title('(a) Density temperature')
     box off
    
    % Right axis
    axes('position',[0.53 0.12 0.4 0.72]);
     plot(h./1000,z./1000,'r','linewidth',2)
     hold on
     plot(h_env./1000,z./1000,'b','linewidth',2)
    
     title('(b) Moist static energy')
     xlabel('h (kJ/kg)')
     set(gca,'xtick',[40 50 60 70 80],'xlim',[50 80])
     set(gca,'yticklabel','')
     box off
     
     l = legend('plume','env');
     set(l,'box','off')
     
    %suptitle(['RH_{env} = ' num2str(RH) ', \epsilon(z) = ' num2str(entrain) '/z'])
    
end


end


function h = calc_MSE(T,qt,p,z)
% Function to calculate moist static energy

% Get constants
global c

% calculate proportions of vapor, liquid and solid
[qv,ql,qi] = calc_saturation(p,T,qt);


% calculate moist static energies of components
hd = c.cp.*(T-c.T0)  + c.g.*z;
hv = c.cpv.*(T-c.T0) + c.g.*z + c.Lv0;
hl = c.cpl.*(T-c.T0) + c.g.*z;
hi = c.cpi.*(T-c.T0) + c.g.*z - (c.Ls0-c.Lv0);

% Calculate moist static energy per unit mass of moist air
h = hd.*(1-qt) + qv.*hv + ql.*hl + qi.*hi;


end

function [fl,fs] = fallout(gamma,ql,qi,dqls,T)
% Function to calculate precipitation fallout terms

e = min(   0,dqls./(1-ql-qi)  );

fT = -gamma.*e;
[fliq,fice] = calc_fice(T);

fl = fT.*fliq;
fs = fT.*fice;

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


function [q,ql,qi] = calc_saturation(p,T,q_t)
% Function to calculate the specific humidities of vapor, liquid and solid
% at saturation


% calculate saturation specific humidity
qs = qq_sat(T,p,q_t);
q = min(q_t,qs);

q(q<0) = 0;

% Divide into liquid and ice
[fliq,fice] = calc_fice(T);

ql = fliq.*(q_t-q);
qi = fice.*(q_t-q);

ql(q_t-qs<0) = 0;
qi(q_t-qs<0) = 0;


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
    
    
    
    
    
    
    



