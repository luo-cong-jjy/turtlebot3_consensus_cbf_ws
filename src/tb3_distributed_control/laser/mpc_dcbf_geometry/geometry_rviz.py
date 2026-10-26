#!/usr/bin/env python3
"""Instantiate the RViz template for the launch-selected robot namespace."""
from pathlib import Path
import subprocess
import tempfile

import yaml


def render_config(template, robot_namespace, target_frame):
    config = yaml.safe_load(Path(template).read_text(encoding='utf-8'))
    manager = config['Visualization Manager']
    manager['Global Options']['Fixed Frame'] = target_frame
    prefix = robot_namespace.strip('/')
    for display in manager['Displays']:
        if display['Class'] == 'rviz/RobotModel':
            display['TF Prefix'] = prefix
            display['Robot Description'] = '/' + prefix + '/robot_description'
        for key in ('Topic', 'Marker Topic'):
            if key in display:
                display[key] = '/' + prefix + '/' + display[key].lstrip('/')
    return config


def main():
    import rospy
    rospy.init_node('geometry_rviz')
    namespace = rospy.get_param('~robot_namespace', 'tb3_0').strip('/')
    config = render_config(rospy.get_param('~config_file'), namespace,
                           rospy.get_param('~target_frame', 'world'))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rviz',
                                     prefix='tb3_laser_geometry_', delete=False) as stream:
        yaml.safe_dump(config, stream, allow_unicode=True)
        filename = stream.name
    process = None
    try:
        process = subprocess.Popen(['rviz', '-d', filename,
                                    '__ns:=/' + namespace, '__name:=geometry_rviz_display'])
        def stop():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
        rospy.on_shutdown(stop)
        return process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
        Path(filename).unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
