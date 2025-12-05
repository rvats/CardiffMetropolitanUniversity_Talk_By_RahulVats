from math import atan, atan2, sin, cos, tan, pi, sqrt, radians, degrees

f = 1/298.257223
R = 6378137
e = 8.1819190842622e-2

def llaToECEF(coords):
    
	coords = [radians(val) for val in coords]
	(lat, lon, h) = coords
	lambds = atan((1-f)**2*tan(lat))
	rs = sqrt(R**2/(1+(1/(1-f)**2-1)*sin(lambds)**2))
	
	x = rs*cos(lambds)*cos(lon) + h*cos(lat)*cos(lon)
	y = rs*cos(lambds)*sin(lon) + h*cos(lat)*cos(lon)
	z = rs*sin(lambds) + h*sin(lat)
	
	return (x,y,z)

def ECEFTolla(coords):
	(x, y, z) = coords
	b = sqrt(pow(R,2) * (1-pow(e,2)))
	ep = sqrt((pow(R,2)-pow(b,2))/pow(b,2))
	p = sqrt(pow(x,2)+pow(y,2))
	th = atan2(R*z, b*p)
	lon = atan2(y, x)
	lat = atan2((z+ep*ep*b*pow(sin(th),3)), (p-e*e*R*pow(cos(th),3)))
	n = R/sqrt(1-e*e*pow(sin(lat),2))
	alt = p/cos(lat)-n
	lat = (lat*180)/pi
	lon = (lon*180)/pi
	
	return (lat, lon, alt)
	
# ecef = llaToECEF(radians(12.5774271), radians(55.7847283), radians(0))
# print ecef
# print ECEFTolla(ecef[0], ecef[1], ecef[2])
