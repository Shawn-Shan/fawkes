#!/bin/sh
#
# This is the default apptemplate error script
#
if ( test -n "$2" ) ; then 
	echo "$1 Error"
	echo "An unexpected error has occurred during execution of the main script"
	echo ""
	echo "$2: $3"
	echo ""
	echo "See the Console for a detailed traceback."
else
	echo "$1 Error"

	# Usage: ERRORURL <anURL> <a button label>, this is used by the 
	# bundle runner to put up a dialog.
	#echo "ERRORURL: http://www.python.org/ Visit the Python Website
# echo "ERRORURL: http://homepages.cwi.nl/~jack/macpython/index.html Visit the MacPython Website"
fi
