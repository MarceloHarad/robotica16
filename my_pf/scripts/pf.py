#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This is the starter code for the robot localization project

    Originally cloned from Paul Ruvolo's CompRobo 15:
    https://github.com/paulruvolo/comprobo15/

"""
import random
import rospy
import chama_mapa

from std_msgs.msg import Header, String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, Vector3
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker, MarkerArray
from copy import deepcopy

import tf
from tf import TransformListener
from tf import TransformBroadcaster
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
from random import gauss

import math
import time

import numpy as np
from numpy.random import random_sample
from sklearn.neighbors import NearestNeighbors
from occupancy_field import OccupancyField

from scipy.stats import norm

from helper_functions import (convert_pose_inverse_transform,
                              convert_translation_rotation_to_pose,
                              convert_pose_to_xy_and_theta,
                              angle_diff)



#Todos msg.header.stamp foram trocados por rospy.Time(0) para que nao ocorre erro de ExtrapolationTime


class Particle(object):
    """ Represents a hypothesis (particle) of the robot's pose consisting of x,y and theta (yaw)
        Attributes:
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized
    """

    def __init__(self,x=0.0,y=0.0,theta=0.0,w=1.0):
        """ Construct a new Particle
            x: the x-coordinate of the hypothesis relative to the map frame
            y: the y-coordinate of the hypothesis relative ot the map frame
            theta: the yaw of the hypothesis relative to the map frame
            w: the particle weight (the class does not ensure that particle weights are normalized """
        self.w = w
        self.theta = theta
        self.x = x
        self.y = y

    def as_pose(self):
        """ A helper function to convert a particle to a geometry_msgs/Pose message """
        orientation_tuple = tf.transformations.quaternion_from_euler(0,0,self.theta)
        return Pose(position=Point(x=self.x,y=self.y,z=0), orientation=Quaternion(x=orientation_tuple[0], y=orientation_tuple[1], z=orientation_tuple[2], w=orientation_tuple[3]))

    # TODO: define additional helper functions if needed

class ParticleFilter:
    """ The class that represents a Particle Filter ROS Node
        Attributes list:
            initialized: a Boolean flag to communicate to other class methods that initializaiton is complete
            base_frame: the name of the robot base coordinate frame (should be "base_link" for most robots)
            map_frame: the name of the map coordinate frame (should be "map" in most cases)
            odom_frame: the name of the odometry coordinate frame (should be "odom" in most cases)
            scan_topic: the name of the scan topic to listen to (should be "scan" in most cases)
            n_particles: the number of particles in the filter
            d_thresh: the amount of linear movement before triggering a filter update
            a_thresh: the amount of angular movement before triggering a filter update
            laser_max_distance: the maximum distance to an obstacle we should use in a likelihood calculation
            pose_listener: a subscriber that listens for new approximate pose estimates (i.e. generated through the rviz GUI)
            particle_pub: a publisher for the particle cloud
            laser_subscriber: listens for new scan data on topic self.scan_topic
            tf_listener: listener for coordinate transforms
            tf_broadcaster: broadcaster for coordinate transforms
            particle_cloud: a list of particles representing a probability distribution over robot poses
            current_odom_xy_theta: the pose of the robot in the odometry frame when the last filter update was performed.
                                   The pose is expressed as a list [x,y,theta] (where theta is the yaw)
            map: the map we will be localizing ourselves in.  The map should be of type nav_msgs/OccupancyGrid
    """
    def __init__(self):
        self.initialized = False        # make sure we don't perform updates before everything is setup
        rospy.init_node('pf')           # tell roscore that we are creating a new node named "pf"

        self.base_frame = "base_link"   # the frame of the robot base
        self.map_frame = "map"          # the name of the map coordinate frame
        self.odom_frame = "odom"        # the name of the odometry coordinate frame
        self.scan_topic = "scan"        # the topic where we will get laser scans from

        self.n_particles = 100          # the number of particles to use

        self.d_thresh = 0.2             # the amount of linear movement before performing an update
        self.a_thresh = math.pi/6       # the amount of angular movement before performing an update

        self.laser_max_distance = 2.0   # maximum penalty to assess in the likelihood field model

        self.sigma = 0.08 

        # TODO: define additional constants if needed

        # Setup pubs and subs

        # pose_listener responds to selection of a new approximate robot location (for instance using rviz)
        self.pose_listener = rospy.Subscriber("initialpose", PoseWithCovarianceStamped, self.update_initial_pose)

        
            
        # publish the current particle cloud.  This enables viewing particles in rviz.
        self.particle_pub = rospy.Publisher("particlecloud", PoseArray, queue_size=10)
        self.marker_pub = rospy.Publisher("markers", MarkerArray, queue_size=10)

        # laser_subscriber listens for data from the lidar
        # Dados do Laser: Mapa de verossimilhança/Occupancy field/Likehood map e Traçado de raios.
        # Traçado de raios: Leitura em ângulo que devolve distância, através do sensor. Dado o mapa,
        # extender a direção da distância pra achar um obstáculo. 
        self.laser_subscriber = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_received)

        # enable listening for and broadcasting coordinate transforms
        #atualização de posição(odometria)
        self.tf_listener = TransformListener()
        self.tf_broadcaster = TransformBroadcaster()

        self.particle_cloud = []

        self.current_odom_xy_theta = []

        #Chamar o mapa a partir de funcao externa
        mapa = chama_mapa.obter_mapa()

        # request the map from the map server, the map should be of type nav_msgs/OccupancyGrid
        # TODO: fill in the appropriate service call here.  The resultant map should be assigned be passed
        #       into the init method for OccupancyField

        # for now we have commented out the occupancy field initialization until you can successfully fetch the map
        self.occupancy_field = OccupancyField(mapa)
        self.initialized = True


    def update_robot_pose(self):
        print("Update Robot Pose")
        """ Update the estimate of the robot's pose given the updated particles.
            There are two logical methods for this:
                (1): compute the mean pose
                (2): compute the most likely pose (i.e. the mode of the distribution)
        """
        # first make sure that the particle weights are normalized
        self.normalize_particles()


        # TODO: assign the lastest pose into self.robot_pose as a geometry_msgs.Pose object
        # just to get started we will fix the robot's pose to always be at the origin
        
        #Variaveis para soma do X,Y e angulo Theta da particula
        x_sum = 0
        y_sum = 0
        theta_sum = 0


        #Loop de soma para as variaveis relativas a probabilidade da particula
        for p in self.particle_cloud:
            x_sum += p.x * p.w
            y_sum += p.y * p.w
            theta_sum += p.theta * p.w

        #Atributo robot_pose eh o pose da melhor particula possivel a partir das variaveis de soma
        self.robot_pose = Particle(x=x_sum, y=y_sum, theta=theta_sum).as_pose()


    def update_particles_with_odom(self,msg):
        print("Update Particles with Odom")
        """ Update the particles using the newly given odometry pose.
            The function computes the value delta which is a tuple (x,y,theta)
            that indicates the change in position and angle between the odometry
            when the particles were last updated and the current odometry.

            msg: this is not really needed to implement this, but is here just in case.
        """
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        # compute the change in x,y,theta since our last update
        if self.current_odom_xy_theta:
            old_odom_xy_theta = self.current_odom_xy_theta
            delta = (new_odom_xy_theta[0] - self.current_odom_xy_theta[0],
                     new_odom_xy_theta[1] - self.current_odom_xy_theta[1],
                     new_odom_xy_theta[2] - self.current_odom_xy_theta[2])

            self.current_odom_xy_theta = new_odom_xy_theta
        else:
            self.current_odom_xy_theta = new_odom_xy_theta


        #R eh o raio feito a partir de um pitagoras com o X e Y da variacao Delta
        r = math.sqrt(delta[0]**2 + delta[1]**2)

        #Funcao para conseguir um valor entre -pi e pi e subtrair o antigo valor de theta. (Achei um pouco confusa, )
        phi = math.atan2(delta[1], delta[0])-old_odom_xy_theta[2]
        
        for particle in self.particle_cloud:
            particle.x += r*math.cos(phi+particle.theta)
            particle.y += r*math.sin(phi+particle.theta)
            particle.theta += delta[2]
    
        # TODO: modify particles using delta
        # For added difficulty: Implement sample_motion_odometry (Prob Rob p 136)

    def map_calc_range(self,x,y,theta):
        """ Difficulty Level 3: implement a ray tracing likelihood model... Let me know if you are interested """
        # TODO: nothing unless you want to try this alternate likelihood model
        pass

    def resample_particles(self):
        """ Resample the particles according to the new particle weights.
            The weights stored with each particle should define the probability that a particular
            particle is selected in the resampling step.  You may want to make use of the given helper
            function draw_random_sample.
        """
        #Primeiro de tudo, normalizar particulas
        self.normalize_particles()

        #Criar array do numpy vazia do tamanho do numero de particulas.
        values = np.empty(self.n_particles)

        #Preencher essa lista com os indices das particulas
        for i in range(self.n_particles):
            values[i] = i

        #Criar uma lista para novas particulas
        new_particles = []

        #Criar lista com os indices das particulas com mais probabilidade
        random_particles = ParticleFilter.weighted_values(values,[p.w for p in self.particle_cloud],self.n_particles)
        for i in random_particles:
            #Transformar o I em inteiro para corrigir bug de float
            int_i = int(i)

            #Pegar particula na possicao I na nuvem de particulas.
            p = self.particle_cloud[int_i]

            #Adicionar particulas somando um valor aleatorio da distribuicao gauss com media = 0 e desvio padrao = 0.025
            new_particles.append(Particle(x=p.x+gauss(0,.025),y=p.y+gauss(0,.025),theta=p.theta+gauss(0,.025)))

        #Igualar nuvem de particulas a novo sample criado
        self.particle_cloud = new_particles
        #Normalizar mais uma vez as particulas.
        self.normalize_particles()

    def update_particles_with_laser(self, msg):
        print("Update Particles with laser")
        """ Updates the particle weights in response to the scan contained in the msg """
        

        for p in self.particle_cloud:
            for i in range(360):
                #Distancia no angulo I
                distancia = msg.ranges[i]

                x = distancia * math.cos(i + p.theta)
                y = distancia * math.sin(i + p.theta)

                #Verificar se distancia minima eh diferente de nan
                erro_nan = self.occupancy_field.get_closest_obstacle_distance(x,y)
                if erro_nan is not float('nan'):
                    # Adicionar valor para corrigir erro de nan (Retirado de outro codigo)
                    p.w += math.exp(-erro_nan*erro_nan/(2*self.sigma**2))


        #Normalizar particulas e criar um novo sample das mesmas
        self.normalize_particles()
        self.resample_particles()

    @staticmethod
    def weighted_values(values, probabilities, size):
        """ Return a random sample of size elements from the set values with the specified probabilities
            values: the values to sample from (numpy.ndarray)
            probabilities: the probability of selecting each element in values (numpy.ndarray)
            size: the number of samples
        """
        bins = np.add.accumulate(probabilities)
        return values[np.digitize(random_sample(size), bins)]

    @staticmethod
    #Nao estou usando esse metodo. Apenas o weighted_values
    def draw_random_sample(choices, probabilities, n):
        print("Draw Random Sample")
        """ Return a random sample of n elements from the set choices with the specified probabilities
            choices: the values to sample from represented as a list
            probabilities: the probability of selecting each element in choices represented as a list
            n: the number of samples
        """
        values = np.array(range(len(choices)))
        probs = np.array(probabilities)
        bins = np.add.accumulate(probs)
        inds = values[np.digitize(random_sample(n), bins)]
        samples = []
        for i in inds:
            samples.append(deepcopy(choices[int(i)]))
        return samples

    def update_initial_pose(self, msg):
        print("Update Initial Pose")
        """ Callback function to handle re-initializing the particle filter based on a pose estimate.
            These pose estimates could be generated by another ROS Node or could come from the rviz GUI """
        xy_theta = convert_pose_to_xy_and_theta(msg.pose.pose)
        self.initialize_particle_cloud(xy_theta)
        self.fix_map_to_odom_transform(msg)

    def initialize_particle_cloud(self, xy_theta=None):
        """ Initialize the particle cloud.
            Arguments
            xy_theta: a triple consisting of the mean x, y, and theta (yaw) to initialize the
                      particle cloud around.  If this input is ommitted, the odometry will be used """
        if xy_theta == None:
            xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)

        self.particle_cloud = []
        # TODO create particles

        # Criando particula
        print("Inicializacao da Cloud de Particulas")

        #Valor auxiliar para nao dar erro na hora de criacao das particulas. Irrelevante
        valor_aux = 0.3
        
        for i in range (self.n_particles):
            self.particle_cloud.append(Particle(0, 0, 0, valor_aux))

        # Randomizar particulas em volta do robo.
        for i in self.particle_cloud:
            i.x = xy_theta[0]+ random.uniform(-1,1)
            i.y = xy_theta[1]+ random.uniform(-1,1)
            i.theta = xy_theta[2]+ random.uniform(-45,45)
        
        #Normalizar as particulas e dar update na posicao do robo
        self.normalize_particles()
        self.update_robot_pose()
        print(xy_theta)


    def normalize_particles(self):
        """ Make sure the particle weights define a valid distribution (i.e. sum to 1.0) """

        #Soma total das probabilidades das particulas
        w_sum = sum([p.w for p in self.particle_cloud])

        #Dividir cada probabilidade de uma particula pela soma total
        for particle in self.particle_cloud:
            particle.w /= w_sum
        # TODO: implement this

    def publish_particles(self, msg):
        print("Publicar Particulas.")
        print(len(self.particle_cloud))
        particles_conv = []
        for p in self.particle_cloud:
            particles_conv.append(p.as_pose())
        # actually send the message so that we can view it in rviz
        self.particle_pub.publish(PoseArray(header=Header(stamp=rospy.Time.now(),
                                            frame_id=self.map_frame),
                                  poses=particles_conv))

    def scan_received(self, msg):
        """ This is the default logic for what to do when processing scan data.
            Feel free to modify this, however, I hope it will provide a good
            guide.  The input msg is an object of type sensor_msgs/LaserScan """
        if not(self.initialized):
            print("Not Initialized")
            # wait for initialization to complete
            return

        if not(self.tf_listener.canTransform(self.base_frame,msg.header.frame_id,rospy.Time(0))):
            print("Not 2")
            # need to know how to transform the laser to the base frame
            # this will be given by either Gazebo or neato_node
            return

        if not(self.tf_listener.canTransform(self.base_frame,self.odom_frame,rospy.Time(0))):
            print("Not 3")
            # need to know how to transform between base and odometric frames
            # this will eventually be published by either Gazebo or neato_node
            return

        # calculate pose of laser relative ot the robot base
        p = PoseStamped(header=Header(stamp=rospy.Time(0),
                                      frame_id=msg.header.frame_id))
        self.laser_pose = self.tf_listener.transformPose(self.base_frame,p)

        # find out where the robot thinks it is based on its odometry
        p = PoseStamped(header=Header(stamp = rospy.Time(0),
                                      frame_id=self.base_frame),
                        pose=Pose())
        self.odom_pose = self.tf_listener.transformPose(self.odom_frame, p)
        # store the the odometry pose in a more convenient format (x,y,theta)
        new_odom_xy_theta = convert_pose_to_xy_and_theta(self.odom_pose.pose)
        print("PASSOU")
        if not(self.particle_cloud):
            # now that we have all of the necessary transforms we can update the particle cloud
            self.initialize_particle_cloud()
            # cache the last odometric pose so we can only update our particle filter if we move more than self.d_thresh or self.a_thresh
            self.current_odom_xy_theta = new_odom_xy_theta
            # update our map to odom transform now that the particles are initialized
            self.fix_map_to_odom_transform(msg)
        elif (math.fabs(new_odom_xy_theta[0] - self.current_odom_xy_theta[0]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[1] - self.current_odom_xy_theta[1]) > self.d_thresh or
              math.fabs(new_odom_xy_theta[2] - self.current_odom_xy_theta[2]) > self.a_thresh):
            # we have moved far enough to do an update!
            self.update_particles_with_odom(msg)    # update based on odometry
            self.update_particles_with_laser(msg)   # update based on laser scan
            self.update_robot_pose()                # update robot's pose
            self.resample_particles()               # resample particles to focus on areas of high density
            self.fix_map_to_odom_transform(msg)     # update map to odom transform now that we have new particles
        # publish particles (so things like rviz can see them)
        self.publish_particles(msg)


    # direcionar particulas quando um obstaculo é identificado.

    def fix_map_to_odom_transform(self, msg):
        print("Fix Map to Odom Transform")
        """ This method constantly updates the offset of the map and
            odometry coordinate systems based on the latest results from
            the localizer """
        (translation, rotation) = convert_pose_inverse_transform(self.robot_pose)
        p = PoseStamped(pose=convert_translation_rotation_to_pose(translation,rotation),
                        header=Header(stamp=rospy.Time(0),frame_id=self.base_frame))
        self.odom_to_map = self.tf_listener.transformPose(self.odom_frame, p)
        (self.translation, self.rotation) = convert_pose_inverse_transform(self.odom_to_map.pose)

    def broadcast_last_transform(self):
        print("Broadcast")
        """ Make sure that we are always broadcasting the last map
            to odom transformation.  This is necessary so things like
            move_base can work properly. """
        if not(hasattr(self,'translation') and hasattr(self,'rotation')):
            return
        self.tf_broadcaster.sendTransform(self.translation,
                                          self.rotation,
                                          rospy.get_rostime(),
                                          self.odom_frame,
                                          self.map_frame)

if __name__ == '__main__':
    n = ParticleFilter()
    r = rospy.Rate(5)


    while not(rospy.is_shutdown()):
        # in the main loop all we do is continuously broadcast the latest map to odom transform
        n.broadcast_last_transform()
        r.sleep()
    