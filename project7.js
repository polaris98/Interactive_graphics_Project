// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.
// This function takes the translation and two rotation angles (in radians) as input arguments.
// The two rotations are applied around x and y axes.
// It returns the combined 4x4 transformation matrix as an array in column-major order.
// You can use the MatrixMult function defined in project5.html to multiply two 4x4 matrices in the same format.


function GetModelViewMatrix( translationX, translationY, translationZ, rotationX, rotationY )
{
	
	 
    // Calculate sine and cosine values
    var cosX = Math.cos(rotationX);
    var sinX = Math.sin(rotationX);
    var cosY = Math.cos(rotationY);
    var sinY = Math.sin(rotationY);

    // Form the rotation matrix
    var rotX = [
        1, 0, 0, 0,
        0, cosX, sinX, 0,
        0, -sinX, cosX, 0,
        0, 0, 0, 1
    ];

    var rotY = [
        cosY, 0, -sinY, 0,
        0, 1, 0, 0,
        sinY, 0, cosY, 0,
        0, 0, 0, 1
    ];

    // Combine rotation matrices
    var rotMat = MatrixMult(rotX, rotY);

    // Form the translation matrix
    var trans = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        translationX, translationY, translationZ, 1
    ];

    // Combine transformation
    return  MatrixMult(trans, rotMat);
}


class MeshDrawer {
    // The constructor is a good place for taking care of the necessary initializations.
    constructor() {
        // vertex shader
        const vs_source = `
		attribute vec3 pos;
		attribute vec2 txtc;
		attribute vec3 normals;
		
		uniform mat4 mvp;
		uniform mat4 mv;
		uniform mat3 normalMatrix;
		uniform vec3 lightDir;
		
		varying vec2 vtexCoord;
		varying vec3 vNormal;

		uniform bool swapYZ;

		void main() {
			vec3 position = pos;
			if (swapYZ) {
				position = vec3(pos.x, pos.z, pos.y);
			}
			gl_Position = mvp * vec4(position, 1.0);
			vtexCoord = txtc;

			// Transform the normal to eye space and normalize
			vNormal = normalize(normalMatrix * normals);
		}
	`;

        const vs = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vs, vs_source);
        gl.compileShader(vs);

        if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(vs));
            gl.deleteShader(vs);
            return;
        }

        // Fragment Shader
		const fs_source = `

			precision highp float;
			uniform sampler2D tex;
			uniform bool showTexture;
			uniform vec3 lightDir;
			
			varying vec2 vtexCoord;
			varying vec3 vNormal;
			
			uniform mat4 mv;
			
			uniform float shininess;

			void main() {
				vec3 diffuseColor = vec3(1.0, 1.0, 1.0);
				vec3 specularColor = vec3(1.0, 1.0, 1.0);

				vec3 texColor = vec3(1.0); // Default color without texture
				if (showTexture) {
					texColor = texture2D(tex, vtexCoord).rgb;
				}

				// Diffuse term
				vec3 normal = normalize(vNormal);
				vec3 light = normalize(lightDir);
				float NdotL = max(dot(normal, light), 0.0);
				vec3 diffuse = diffuseColor * texColor * NdotL;

				
				// Specular term
				vec3 viewDir = normalize(-mv[3].xyz);
				vec3 halfwayDir = normalize(lightDir + viewDir);
				float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);
				vec3 specular = specularColor * spec;

				vec3 finalColor = diffuse + specular;
				gl_FragColor = vec4(finalColor, 1.0);
			}

        `;

        const fs = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fs, fs_source);
        gl.compileShader(fs);

        if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
            alert(gl.getShaderInfoLog(fs));
            gl.deleteShader(fs);
            return;
        }

        this.prog = gl.createProgram();
        gl.attachShader(this.prog, vs);
        gl.attachShader(this.prog, fs);
        gl.linkProgram(this.prog);

        if (!gl.getProgramParameter(this.prog, gl.LINK_STATUS)) {
            alert(gl.getProgramInfoLog(this.prog));
            return;
        }

        // Create buffers
        this.position_buffer = gl.createBuffer();
        this.texture_buffer = gl.createBuffer();
        this.normal_buffer = gl.createBuffer();
    }
	
	// This method is called every time the user opens an OBJ file.
	// The arguments of this function is an array of 3D vertex positions,
	// an array of 2D texture coordinates, and an array of vertex normals.
	// Every item in these arrays is a floating point value, representing one
	// coordinate of the vertex position or texture coordinate.
	// Every three consecutive elements in the vertPos array forms one vertex
	// position and every three consecutive vertex positions form a triangle.
	// Similarly, every two consecutive elements in the texCoords array
	// form the texture coordinate of a vertex and every three consecutive 
	// elements in the normals array form a vertex normal.
	// Note that this method can be called multiple times.
	setMesh( vertPos, texCoords, normals)
	{
		
		// [TO-DO] Update the contents of the vertex buffer objects.
		this.numTriangles = vertPos.length / 3;
		
		//set position buffer
		gl.bindBuffer(gl.ARRAY_BUFFER,this.position_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertPos), gl.STATIC_DRAW);
		
		//set vertex buffer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.texture_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);

		//set normals buffer
		gl.bindBuffer(gl.ARRAY_BUFFER, this.normal_buffer);
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

	}
	
	// This method is called when the user changes the state of the
	// "Swap Y-Z Axes" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	swapYZ( swap )
	{
		// [TO-DO] Set the uniform parameter(s) of the vertex shader
		gl.useProgram(this.prog);
		const swapYZLoc = gl.getUniformLocation(this.prog, 'swapYZ');
		gl.uniform1i(swapYZLoc, swap ? 1 : 0);
	}
	
	// This method is called to draw the triangular mesh.
	// The arguments are the model-view-projection transformation matrixMVP,
	// the model-view transformation matrixMV, the same matrix returned
	// by the GetModelViewProjection function above, and the normal
	// transformation matrix, which is the inverse-transpose of matrixMV.
	draw(matrixMVP, matrixMV, matrixNormal) {
		// Clear the color and depth buffers
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
		
		// Use the shader program
		gl.useProgram(this.prog);
		
		// Set the MVP matrix uniform
		const mvpLocation = gl.getUniformLocation(this.prog, 'mvp');
		gl.uniformMatrix4fv(mvpLocation, false, matrixMVP);
		
		// Set the model-view matrix uniform
		const mvLocation = gl.getUniformLocation(this.prog, 'mv');
		gl.uniformMatrix4fv(mvLocation, false, matrixMV);
		
		// Set the normal matrix uniform
		const normalLocation = gl.getUniformLocation(this.prog, 'normalMatrix');
		gl.uniformMatrix3fv(normalLocation, false, matrixNormal);
	
		// Enable and specify vertex position attribute
		const posAttribLocation = gl.getAttribLocation(this.prog, 'pos');
		gl.enableVertexAttribArray(posAttribLocation);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.position_buffer);
		gl.vertexAttribPointer(posAttribLocation, 3, gl.FLOAT, false, 0, 0);

		const normAttribLocation = gl.getAttribLocation(this.prog, 'normals');
		gl.enableVertexAttribArray(normAttribLocation);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.normal_buffer);
		gl.vertexAttribPointer(normAttribLocation, 3, gl.FLOAT, false, 0, 0);
		
		// Enable and specify texture coordinate attribute
		const texCoordAttribLocation = gl.getAttribLocation(this.prog, 'txtc');
		gl.enableVertexAttribArray(texCoordAttribLocation);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.texture_buffer);
		gl.vertexAttribPointer(texCoordAttribLocation, 2, gl.FLOAT, false, 0, 0);
	
		// Draw the triangles
		gl.drawArrays(gl.TRIANGLES, 0, this.numTriangles);
	}
	
	
	// This method is called to set the texture of the mesh.
	// The argument is an HTML IMG element containing the texture data.
	setTexture( img )
	{// [TO-DO] Bind the texture
		const mytex = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, mytex);
		
		// You can set the texture image data using the following command.
		gl.texImage2D( gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img );
		gl.generateMipmap(gl.TEXTURE_2D);
		
		gl.activeTexture(gl.TEXTURE0);
		gl.bindTexture(gl.TEXTURE_2D,mytex);
		
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
		

		// [TO-DO] Now that we have a texture, it might be a good idea to set
		// some uniform parameter(s) of the fragment shader, so that it uses the texture.
		const sampler = gl.getUniformLocation(this.prog, 'tex');
		gl.useProgram(this.prog);
		gl.uniform1i(sampler, 0); 	
	}
	
	// This method is called when the user changes the state of the
	// "Show Texture" checkbox. 
	// The argument is a boolean that indicates if the checkbox is checked.
	showTexture( show )
	{
		// [TO-DO] set the uniform parameter(s) of the fragment shader to specify if it should use the texture.
		const showTexLocation = gl.getUniformLocation(this.prog, 'showTexture');
		gl.useProgram(this.prog);
		gl.uniform1i(showTexLocation, show ? 1 : 0);
	}
	
	// This method is called to set the incoming light direction
	setLightDir( x, y, z )
	{
        // Set the uniform parameter(s) of the fragment shader to specify the light direction.
        const lightDirLocation = gl.getUniformLocation(this.prog, 'lightDir');
        gl.useProgram(this.prog);
        gl.uniform3f(lightDirLocation,x,y,z);
    
	}
	
	// This method is called to set the shininess of the material
	setShininess( shininess )
	{
		// [TO-DO] set the uniform parameter(s) of the fragment shader to specify the shininess.
		// Set uniform parameter for shininess
		const shininessLocation = gl.getUniformLocation(this.prog, 'shininess');
		gl.useProgram(this.prog);
		gl.uniform1f(shininessLocation, shininess);
	}
}


// This function is called for every step of the simulation.
// Its job is to advance the simulation for the given time step duration dt.
// It updates the given positions and velocities.
function SimTimeStep(dt, positions, velocities, springs, stiffness, damping, particleMass, gravity, restitution, pinned0, pinned1 ) {
    var forces = Array(positions.length); // Initialize forces to zero

    // Compute forces
	for (var i = 0; i < positions.length; i++) {
        forces[i] = new Vec3(0, 0, 0);
        // Compute gravity force
        forces[i].inc(gravity.mul(particleMass));
    }

    // for (var i = 0; i < springs.length; i++) {
	// 	var spring = springs[i];
	// 	var particle0 = spring.p0;
	// 	var particle1 = spring.p1;
		
	// 	//Calculate sping force
    //     var d = positions[particle1].sub(positions[particle0]);
    //     var dl = d.len();
    //     var springForce = d.mul(stiffness * (dl - spring.rest) / dl);

    //     // Calculate the damping force
	// 	var delta_v =  velocities[particle1].sub(velocities[particle0]);
    //     var dampingForce = d.mul(damping * delta_v.dot(d) / dl);

    //     // Total force 
    //     var totalForce = springForce.add(dampingForce);

    //     // Apply forces to the particles
    //     forces[spring.p0] = forces[spring.p0].add(totalForce);
    //     forces[spring.p1] = forces[spring.p1].sub(totalForce);
    // }

	 // Update positions and velocities using semi-implicit Euler's method
	 for (var i = 0; i < positions.length; i++) {
        if (!particleMass || i==pinned0 || i==pinned1) continue;
        velocities[i].inc(forces[i].mul(dt).div(particleMass)); // Update velocity
        positions[i].inc(velocities[i].mul(dt)); // Update position
    }

	//apply dumping and sleeping
	for(var i = 0;i<velocities.length;i++) {
		if (!particleMass) continue;
		// damping
		velocities[i].scale(1 - damping * dt);
		// sleeping
		if (velocities[i].len < 0.01)
		   velocities[i].set(0, 0, 0);
	}

	for (var i = 0; i < positions.length; i++) {
        // Check each dimension to see if the particle is outside the box
        ['x', 'y', 'z'].forEach(dim => {
            if (positions[i][dim] < -1) {
                positions[i][dim] = -1 + (-1 - positions[i][dim]); // Reflect position
                velocities[i][dim] = -velocities[i][dim] * restitution; // Reflect velocity
            } else if (positions[i][dim] > 1) {
                positions[i][dim] = 1 - (positions[i][dim] - 1); // Reflect position
                velocities[i][dim] = -velocities[i][dim] * restitution; // Reflect velocity
            }
        });
    }

}

