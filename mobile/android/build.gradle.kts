plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.2.20" apply false
}

val buildRoot = providers.gradleProperty("neshter.buildRoot").get()
allprojects { layout.buildDirectory.set(file("$buildRoot/${project.name}")) }
