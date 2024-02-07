#!/bin/bash
rm -rf cdk.* package* node_modules/
npm install aws-cdk-lib
. ~/.bash_profile
cdk bootstrap aws://$AWS_ACCOUNT_ID/$AWS_REGION
npm install
cdk destroy --app "npx ts-node --prefer-ts-exts ./pipeline.ts"  
