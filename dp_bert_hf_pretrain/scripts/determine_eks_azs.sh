#!/bin/bash
# Determine appropriate EKS AZs depending on default AWS region

REGION_CODE=$(aws configure get region)

if [[ $REGION_CODE == "us-east-1" ]]
then
	AZ1="use1-az6"
	AZ2="use1-az5"
elif [[ $REGION_CODE == "us-west-2" ]]
then
	AZ1="usw2-az4"
	AZ2="usw2-az3"
else
	echo "Please set your default AWS region to us-east-1 or us-west-2 using 'aws configure' and re-run this script"
	exit 1
fi

EKSAZ1=$(aws ec2 describe-availability-zones \
--region $REGION_CODE \
--query "AvailabilityZones[]" \
--filters "Name=zone-id,Values=$AZ1" \
--query "AvailabilityZones[].ZoneName" \
--output text)

EKSAZ2=$(aws ec2 describe-availability-zones \
--region $REGION_CODE \
--query "AvailabilityZones[]" \
--filters "Name=zone-id,Values=$AZ2" \
--query "AvailabilityZones[].ZoneName" \
--output text)

echo -e "\nYour EKS availability zones are $EKSAZ1 and $EKSAZ2\n"

# Save EKS AZs for use by subsequent scripts
mkdir -p /tmp/ekstut/
cat <<EOF > /tmp/ekstut/eks_azs.sh
EKSAZ1=$EKSAZ1
EKSAZ2=$EKSAZ2
EOF
