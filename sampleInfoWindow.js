 function setMarkers(branches, map) {
    var bounds = new google.maps.LatLngBounds();
    var contentString = null;
    var infowindow = null;
    infowindow = new google.maps.InfoWindow();
    for (var i = 0; i < branches.length; i++) {
        var marker = null;
        branch = branches[i];
        var myLatlngMarker = new google.maps.LatLng(branch[0], branch[1]);
        contentString = '<p>' + branch[3] + '</p>';

        var marker = new google.maps.Marker({
            position: myLatlngMarker,
            map: map,
            title: branch[2]
        });

        google.maps.event.addListener(marker, 'click', function(content) {
            return function(){
                infowindow.setContent(content);//set the content
                infowindow.open(map,this);
            }
        }(contentString));

        bounds.extend(myLatlngMarker);
    }

    map.fitBounds(bounds);
}