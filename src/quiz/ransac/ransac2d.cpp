/* \author Aaron Brown */
// Quiz on implementing simple RANSAC line fitting

#include "../../render/render.h"
#include <unordered_set>
#include "../../processPointClouds.h"
// using templates for processPointClouds so also include .cpp to help linker
#include "../../processPointClouds.cpp"

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
  	// Add inliers
  	float scatter = 0.6;
  	for(int i = -5; i < 5; i++)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = i+scatter*rx;
  		point.y = i+scatter*ry;
  		point.z = 0;

  		cloud->points.push_back(point);
  	}
  	// Add outliers
  	int numOutliers = 10;
  	while(numOutliers--)
  	{
  		double rx = 2*(((double) rand() / (RAND_MAX))-0.5);
  		double ry = 2*(((double) rand() / (RAND_MAX))-0.5);
  		pcl::PointXYZ point;
  		point.x = 5*rx;
  		point.y = 5*ry;
  		point.z = 0;

  		cloud->points.push_back(point);

  	}
  	cloud->width = cloud->points.size();
  	cloud->height = 1;

  	return cloud;

}

pcl::PointCloud<pcl::PointXYZ>::Ptr CreateData3D()
{
	ProcessPointClouds<pcl::PointXYZ> pointProcessor;
	return pointProcessor.loadPcd("../../../sensors/data/pcd/simpleHighway.pcd");
}


pcl::visualization::PCLVisualizer::Ptr initScene()
{
	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer ("2D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
  	viewer->initCameraParameters();
  	viewer->setCameraPosition(0, 0, 15, 0, 1, 0);
  	viewer->addCoordinateSystem (1.0);
  	return viewer;
}

//MyAnswer
std::unordered_set<int> RansacLine2(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	
	while(maxIterations-- > 0 )
	{
		// Randomly pick two points
		pcl::PointXYZ point1 = cloud->points.at(rand() % (cloud->points.size()));
		pcl::PointXYZ point2 = cloud->points.at(rand() % (cloud->points.size()));
		
		float a = point1.y - point2.y;
		float b = point2.x - point1.x;
		float c = point1.x*point2.y - point2.x*point1.y;

		std::unordered_set<int> inliers;
		for(auto index = cloud->points.begin(); index != cloud->points.end(); index++)
		{
			float d = fabs(a*(*index).x+b*(*index).y+c) / sqrt(a*a+b*b);

			if(d <= distanceTol)
				inliers.insert(index-cloud->begin());

		}

		if(inliers.size()>inliersResult.size())
		{
			inliersResult = inliers;
		}

	}	
	return inliersResult;

}

// Udacity
std::unordered_set<int> RansacLine1(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();

	std::unordered_set<int> inliersResult;
	srand(time(NULL));
	
	// TODO: Fill in this function

	// For max iterations 

	// Randomly sample subset and fit line

	// Measure distance between every point and fitted line
	// If distance is smaller than threshold count it as inlier

	// Return indicies of inliers from fitted line with most inliers

	while(maxIterations--)
	{

		// Randomly pick two points

		std::unordered_set<int> inliers;
		while (inliers.size() < 2)
			inliers.insert(rand()%(cloud->points.size()));
		
		float x1, y1, x2, y2;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;

		float a = (y1-y2);
		float b = (x2-x1);
		float c = (x1*y2-x2*y1);

		for(int index = 0; index < cloud->points.size(); index++)
		{
			
			if(inliers.count(index)>0)
				continue;
			
			pcl::PointXYZ point = cloud->points[index];
			float x3 = point.x;
			float y3 = point.y;

			float d = fabs(a*x3+b*y3+c)/sqrt(a*a+b*b);

			if(d <= distanceTol)
				inliers.insert(index);

		}

		if(inliers.size()>inliersResult.size())
		{
			inliersResult = inliers;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "Ransac took" << "milliseconds" << std::endl;
	
	return inliersResult;

}

std::unordered_set<int> RansacPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
    std::unordered_set<int> inliersResult;
    srand(time(NULL));

    while (maxIterations-- > 0) {
        // Randomly sample subset
        pcl::PointXYZ point1 = cloud->points.at(rand() % (cloud->points.size()));
        pcl::PointXYZ point2 = cloud->points.at(rand() % (cloud->points.size()));
        pcl::PointXYZ point3 = cloud->points.at(rand() % (cloud->points.size()));

        float a = (point2.y - point1.y) * (point3.z - point1.z) - (point2.z - point1.z) * (point3.y - point1.y);
        float b = (point2.z - point1.z) * (point3.x - point1.x) - (point2.x - point1.x) * (point3.z - point1.z);
        float c = (point2.x - point1.x) * (point3.y - point1.y) - (point2.y - point1.y) * (point3.x - point1.x);
        float d = -1 * (a * point1.x + b * point1.y + c * point1.z);

        std::unordered_set<int> inliers;
        for (auto index = cloud->points.begin(); index != cloud->points.end(); ++index) {
            float d = fabs(a * (*index).x + b * (*index).y + c * (*index).z + d) / sqrt(a*a + b*b + c*c);
            if (d <= distanceTol) {
                inliers.insert(index - cloud->begin());
            }
        }
        if (inliers.size() > inliersResult.size()) {
            inliersResult = inliers;
        }
    }
    return inliersResult;
}

/*
std::unordered_set<int> RansacPlane_SegmentationError(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int maxIterations, float distanceTol)
{
	// Time segmentation process
	auto startTime = std::chrono::steady_clock::now();

	std::unordered_set<int> inliersResult;
	srand(time(NULL));

	while(maxIterations--)
	{

		// Randomly pick two points
		std::unordered_set<int> inliers;
		while (inliers.size() < 2)
			inliers.insert(rand()%(cloud->points.size()));
		
		float x1, y1, z1, x2, y2, z2, x3, y3, z3;

		auto itr = inliers.begin();
		x1 = cloud->points[*itr].x;
		y1 = cloud->points[*itr].y;
		z1 = cloud->points[*itr].z;
		itr++;
		x2 = cloud->points[*itr].x;
		y2 = cloud->points[*itr].y;
		z2 = cloud->points[*itr].z;
		itr++;
		x3 = cloud->points[*itr].x;
		y3 = cloud->points[*itr].y;
		z3 = cloud->points[*itr].z;

		float a = (y2-y1)*(z3-z1)-(z2-z1)*(y3-y1);
		float b = (z2-z1)*(x3-x1)-(x2-x1)*(z3-z1);
		float c = (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
		float d = (a*x1+b*y1+c*z1)*(-1);

		for(int index = 0; index < cloud->points.size(); index++)
		{
			
			if(inliers.count(index)>0)
				continue;
			
			pcl::PointXYZ point = cloud->points[index];
			float x4 = point.x;
			float y4 = point.y;
			float z4 = point.z;

			float d_plane = fabs(a*x4+b*y4+c*z4+d)/sqrt(a*a+b*b+c*c);

			if(d_plane <= distanceTol)
				inliers.insert(index);

		}

		if(inliers.size()>inliersResult.size())
		{
			inliersResult = inliers;
		}

	}

	auto endTime = std::chrono::steady_clock::now();
	auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
	std::cout << "Ransac took" << "milliseconds" << std::endl;
	
	return inliersResult;

}
*/

int main ()
{

	// Create viewer
	pcl::visualization::PCLVisualizer::Ptr viewer = initScene();

	// Create data
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = CreateData3D();
	

	// TODO: Change the max iteration and distance tolerance arguments for Ransac function
	std::unordered_set<int> inliers = RansacPlane(cloud, 10, 0.2);

	pcl::PointCloud<pcl::PointXYZ>::Ptr  cloudInliers(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOutliers(new pcl::PointCloud<pcl::PointXYZ>());

	for(int index = 0; index < cloud->points.size(); index++)
	{
		pcl::PointXYZ point = cloud->points[index];
		if(inliers.count(index))
			cloudInliers->points.push_back(point);
		else
			cloudOutliers->points.push_back(point);
	}


	// Render 2D point cloud with inliers and outliers
	if(inliers.size())
	{
		renderPointCloud(viewer,cloudInliers,"inliers",Color(0,1,0));
  		renderPointCloud(viewer,cloudOutliers,"outliers",Color(1,0,0));
	}
  	else
  	{
  		renderPointCloud(viewer,cloud,"data");
  	}
	
  	while (!viewer->wasStopped ())
  	{
  	  viewer->spinOnce ();
  	}
  	
}
